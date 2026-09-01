#include <Servo.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <LittleFS.h>
#include <math.h> 
#include <vector>
#include <algorithm>
#include <tuple> // std::tupleを使用するために必要
#include <cstdlib>   // rand(), srand()
#include "agent_config.h"
#include "ServerUtils.h"
#include "WiFiManager.h"
#include "calculateTrimmedMean.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// agent_id: 不変なのでRAMで持つだけでOK (送信時にのみ使用)
int agent_id;

// WiFi設定
const char* ssid = "Buffalo-G-7EF4";
const char* password = "76533631";
//const char* ssid = "Buffalo-G-4510";
//const char* password = "33354682";

// UDP設定
IPAddress serverIP(192, 168, 13, 98);
unsigned int serverPort = 5000; // 最初はメインポートに接続
unsigned int agentPort; // エージェント専用ポート
WiFiUDP udp;

// ピン設定
const int digitalInputPin = 4;  // ボタン
const int analogPin1 = 28;
const int analogPin2 = 27;
const int inaSdaPin = 6;
const int inaSclPin = 7;

Servo myServo;

// 1レコード7バイトの圧縮構造体 (RAM保持用 - 明示的1バイトアライメント)
#pragma pack(push, 1)
struct CompressedLogData {
  uint8_t micros24_0;  // タイムスタンプ下位8bit
  uint8_t micros24_1;  // タイムスタンプ中位8bit
  uint8_t micros24_2;  // タイムスタンプ上位8bit
  uint8_t analog0;     // 1バイト: 位相 (phi)
  uint8_t analog1;     // 1バイト: 電力 (INA226)
  uint8_t analog2;     // 1バイト: 曲げセンサ (GP27)
  uint8_t analog3;     // 1バイト: 光センサ (GP28)
};
#pragma pack(pop)

#define CONTROL_PERIOD_US 2000 // 制御周期 (μs)
#define LOG_BUFFER_SIZE   20000 // 20000 * 7バイト = 140,000バイト (約140KB)
CompressedLogData logBuffer[LOG_BUFFER_SIZE];
int logIndex = 0;
bool paused = false;
bool lastButtonState = false;

// START受信時のログ開始時刻
unsigned long startLoggingMillis = 0;
unsigned long startLoggingMicros = 0;
float t_delay;

unsigned long prevLoopEndTime = 0;
unsigned long prevLoopEndTime2 = 0;
float phi = 0;
float omega = 3.0f * 3.14f;
float kappa = 1.0f;  // フィードバックゲイン
float kappa_init = 0.0f;
float kappa_now = 0.0f;
const int PRC_MAX_HARMONICS = 10;
int prcHarmonics = 1;
float prcCosCoeffs[PRC_MAX_HARMONICS + 1] = {0.0f};
float prcSinCoeffs[PRC_MAX_HARMONICS + 1] = {0.0f};
bool bufferOverflowed = false;
float wait_max = 2.0f * M_PI / omega;
float previousFlex = 0.0f; // 前回のフレックスセンサ値

// サーボ制御用パラメータ (サーバーから受信)
float servoCenter = 110.0f;    // サーボ中心角度のデフォルト値
float servoAmplitude = 60.0f; // サーボ振幅のデフォルト値

// 光センサフィードバック用パラメータ
float lightTauSec = 1.0f;             // 光センサ一次遅れフィルタの時定数 [s]
float lightFeedbackGain = 1.0f;       // 光センサによる振幅減衰ゲイン (0.0〜1.0)
float lightFeedbackOmegaGain = 0.0f;  // 光センサによる自然周波数フィードバックゲイン (正: 周波数低下, 負: 周波数増加)
float minAmplitudeRatio = 0.1f;        // 最小振幅倍率 (元の振幅の10%を下限とする)

float feedbackTauSec = 1.0f;           // (その他/元々の) 一次遅れフィルタの時定数 [s]
float filteredLightRaw = 0.0f;         // フィルタ処理後の光センサ値
bool lightFilterInitialized = false;   // フィルタ初期化フラグ

// 初期位相設定 (サーバーから取得)
float initialPhase = 0.0f;             // 指定初期位相 [rad]
bool isRandomInitialPhase = true;      // ランダムモードフラグ

void applyInitialPhase() {
  if (isRandomInitialPhase) {
    phi = (rand() / (float)RAND_MAX) * 2.0f * (float)M_PI;
    Serial.printf("[INFO] Applied random initial phase: phi=%.3f rad\n", phi);
  } else {
    phi = initialPhase;
    Serial.printf("[INFO] Applied specified initial phase: phi=%.3f rad\n", phi);
  }
}


// 停止制御用パラメータ (サーバーから受信)
int stopAgentId = 0; // 停止対象のエージェントID (0は特殊な意味を持つ場合など)
int stopDelaySeconds = 0; // 停止までの秒数

// データ保存間隔を設定 (例: 5ループごとに保存)
const int saveInterval = 5;
int loopCounter = 0;

unsigned long lastRequestTime = 0;  // 最後にリクエストを送信した時刻

// 窓サイズを定義
const int windowSize = 1000; // 必要なサイズに変更
std::vector<int> raw2Window(windowSize, 0); // 固定サイズのリングバッファ
int raw2Index = 0; // 現在のインデックスを管理

// INA226設定
uint8_t ina226Addr = 0x40;
const float SHUNT_OHMS = 0.056f;
const float POWER_W_TO_RAW = 1000.0f; // raw1はmW相当で0..4095に収める
const int POWER_RAW_MAX = 4095;
const unsigned long inaReadErrorLogIntervalMs = 1000;
bool inaReady = false;
unsigned long lastInaReadErrorLogMs = 0;
int lastPowerRaw = 0;
float lastBusVoltV = 0.0f;

// 電力積算用グローバル変数 (10ms区間内の約2msごとの台形則積分)
float powerEnergyJAccum = 0.0f;       // 区間内積算エネルギー [J]
float powerWindowDurationSec = 0.0f;   // 区間内累積時間 [s]
float previousPowerW = 0.0f;          // 前回の有効測定電力 [W]
unsigned long prevInaTimeMicros = 0;  // 前回の有効測定時刻 [micros()]
bool powerAccumValid = false;         // 区間積算が有効か

void resetPowerAccumulator() {
  powerEnergyJAccum = 0.0f;
  powerWindowDurationSec = 0.0f;
  previousPowerW = 0.0f;
  prevInaTimeMicros = 0;
  powerAccumValid = false;
}

bool inaWriteReg16(uint8_t reg, uint16_t value) {
  Wire1.beginTransmission(ina226Addr);
  Wire1.write(reg);
  Wire1.write((uint8_t)(value >> 8));
  Wire1.write((uint8_t)(value & 0xFF));
  return Wire1.endTransmission() == 0;
}

bool inaReadReg16(uint8_t reg, uint16_t &value) {
  Wire1.beginTransmission(ina226Addr);
  Wire1.write(reg);
  if (Wire1.endTransmission(false) != 0) {
    return false;
  }

  if (Wire1.requestFrom((int)ina226Addr, 2) != 2) {
    return false;
  }

  value = ((uint16_t)Wire1.read() << 8) | (uint16_t)Wire1.read();
  return true;
}

bool initIna226() {
  // 平均1回 (AVG=1), バス/シャント変換140us (VBUS CT=000, VSH CT=000), 連続変換 (Mode=111) -> 0x0007
  uint16_t configVal = 0x0007;
  if (!inaWriteReg16(0x00, configVal)) {
    Serial.println("[INA226] ERROR: Failed to write config reg 0x00");
    return false;
  }

  uint16_t readBack = 0;
  if (!inaReadReg16(0x00, readBack)) {
    Serial.println("[INA226] ERROR: Failed to read back config reg 0x00");
    return false;
  }

  if (readBack != configVal) {
    Serial.printf("[INA226] ERROR: Config mismatch! Expected 0x%04X, got 0x%04X\n", configVal, readBack);
    return false;
  }

  Serial.println("[INFO] INA226 initialized & config reg 0x00 verified (0x0007)");
  return true;
}

bool detectIna226Address() {
  Serial.println("[INA226] scanning I2C1 addresses 0x40-0x4F...");
  for (uint8_t addr = 0x40; addr <= 0x4F; addr++) {
    Wire1.beginTransmission(addr);
    uint8_t err = Wire1.endTransmission();
    if (err == 0) {
      ina226Addr = addr;
      Serial.print("[INA226] device found at 0x");
      Serial.println(ina226Addr, HEX);
      return true;
    }
  }
  Serial.println("[INA226] no device found in 0x40-0x4F");
  return false;
}

bool readIna226Measurement(float &currentmA, float &busVoltV) {
  uint16_t shuntRawU16 = 0;
  uint16_t busRawU16 = 0;

  if (!inaReadReg16(0x01, shuntRawU16) || !inaReadReg16(0x02, busRawU16)) {
    unsigned long nowMs = millis();
    if (nowMs - lastInaReadErrorLogMs >= inaReadErrorLogIntervalMs) {
      lastInaReadErrorLogMs = nowMs;
      Serial.println("[INA226] read error (check wiring/pull-up/address)");
    }
    return false;
  }

  int16_t shuntRaw = (int16_t)shuntRawU16;
  float shuntVoltV = (float)shuntRaw * 2.5e-6f;
  float currentA = shuntVoltV / SHUNT_OHMS;
  currentmA = currentA * 1000.0f;
  busVoltV = (float)busRawU16 * 1.25e-3f;

  return true;
}

bool readIna226PowerW(float &powerW) {
  if (!inaReady) {
    return false;
  }

  float currentmA = 0.0f;
  float busVoltV = 0.0f;
  if (!readIna226Measurement(currentmA, busVoltV)) {
    return false;
  }

  lastBusVoltV = busVoltV;
  float pW = (currentmA / 1000.0f) * busVoltV;
  if (pW < 0.0f) {
    pW = 0.0f; // 負電力は0にクランプ
  }
  powerW = pW;
  return true;
}

int encodePowerRaw(float currentmA, float busVoltV) {
  float powerW = (currentmA / 1000.0f) * busVoltV;
  if (powerW <= 0.0f) {
    return 0;
  }

  int powerRaw = (int)(powerW * POWER_W_TO_RAW + 0.5f);
  if (powerRaw > POWER_RAW_MAX) {
    return POWER_RAW_MAX;
  }
  return powerRaw;
}

int readPowerRaw() {
  if (!inaReady) {
    return lastPowerRaw;
  }

  float currentmA = 0.0f;
  float busVoltV = 0.0f;
  if (readIna226Measurement(currentmA, busVoltV)) {
    lastBusVoltV = busVoltV;
    lastPowerRaw = encodePowerRaw(currentmA, busVoltV);
  }
  return lastPowerRaw;
}

float readMonitorVoltageV() {
  if (!inaReady) {
    return lastBusVoltV;
  }

  float currentmA = 0.0f;
  float busVoltV = 0.0f;
  if (readIna226Measurement(currentmA, busVoltV)) {
    lastBusVoltV = busVoltV;
    lastPowerRaw = encodePowerRaw(currentmA, busVoltV);
  }
  return lastBusVoltV;
}

// 正規化する関数
float normalize(float value, float lower, float upper) {
  // 正規化
  float normalized = (value - lower) / (upper - lower) - 0.5f;

  // ±0.5にクリップ
  if (normalized > 0.5f) {
    normalized = 0.5f;
  } else if (normalized < -0.5f) {
    normalized = -0.5f;
  }

  return normalized;
}

float evaluatePRC(float psi) {
  float z = prcCosCoeffs[0];
  int nMax = prcHarmonics;
  if (nMax > PRC_MAX_HARMONICS) {
    nMax = PRC_MAX_HARMONICS;
  }

  for (int n = 1; n <= nMax; n++) {
    float npsi = (float)n * psi;
    z += prcCosCoeffs[n] * cosf(npsi) + prcSinCoeffs[n] * sinf(npsi);
  }

  return z;
}

// ---------------------------------------------------
// 送信バッファをまとめてUDP送信
//   (各パケット先頭に agent_id の1バイトと送信時の時刻4バイトを付加して送る)
// ---------------------------------------------------
void sendLogBuffer() {
  const int maxPacketBytes = 512;
  uint8_t packet[maxPacketBytes];
  const int maxRetries = 100;

  int sentCount = 0;
  int i = 0;

  while (i < logIndex) {
    int retry = 0;
    bool ackReceived = false;

    while (retry < maxRetries && !ackReceived) {
      size_t offset = 0;
      int startIndex = i;

      // サーバー準備チェック
      while (!isServerReady(udp, serverIP, serverPort)) {
        Serial.println("[ERROR] Server not ready. Retrying in 1 second...");
        delay(500);
        if (WiFi.status() != WL_CONNECTED) {
          connectToWiFi(ssid, password);
        }
      }

      // 1) agent_id (1バイト)
      packet[offset++] = (uint8_t)agent_id;

      // 2) 送信時刻 (4バイト)
      uint32_t sendMicros = micros();
      memcpy(&packet[offset], &sendMicros, sizeof(sendMicros));
      offset += sizeof(sendMicros);  // 4バイト

      // 3) データパック詰め
      int perPacketCount = 0;
      uint32_t lastMicros24 = 0;

      while (i < logIndex) {
        if (offset + sizeof(CompressedLogData) > maxPacketBytes) {
          break;
        }

        uint32_t micros24Value = (uint32_t)logBuffer[i].micros24_0 |
                                ((uint32_t)logBuffer[i].micros24_1 << 8) |
                                ((uint32_t)logBuffer[i].micros24_2 << 16);

        memcpy(&packet[offset], &logBuffer[i], sizeof(CompressedLogData));
        offset += sizeof(CompressedLogData);

        lastMicros24 = micros24Value;  // 最後の値を保存
        i++;
        perPacketCount++;
      }

      // 4) UDP送信
      udp.beginPacket(serverIP, serverPort);
      udp.write(packet, offset);
      udp.endPacket();

      Serial.printf("[INFO] Packet sent (%d records). Waiting for ACK...\n", perPacketCount);

      // 5) ACK待機
      ackReceived = waitForAck(udp, agent_id, lastMicros24, 1000);
      if (!ackReceived) {
        retry++;
        Serial.printf("[WARN] ACK not received (retry %d/%d). Resending...\n", retry, maxRetries);
        i = startIndex;  // 再送時は戻る
        delay(100);
      } else {
        sentCount += perPacketCount;
      }
    }

    if (!ackReceived) {
      Serial.println("[ERROR] Failed to receive ACK after multiple retries. Aborting this packet.");
    }
  }

  Serial.printf("[INFO] Sent %d records from RAM (with ACK)\n", sentCount);

  if (bufferOverflowed) {
    Serial.println("[WARN] Some data may have been lost due to buffer overflow.");
    bufferOverflowed = false;
  }
}

// ---------------------------------------------------
// センサ読み取り＋RAMバッファ保存（dtはサーボ用のみ）
// ---------------------------------------------------
void logSensorData() {
  unsigned long now = micros();
  unsigned long dt = now - prevLoopEndTime;
  unsigned long elapsed = now - startLoggingMicros;
  prevLoopEndTime = now;

  // INA226からW単位の量子化前電力値を取得し、台形則で積算
  float currentPowerW = 0.0f;
  bool inaSuccess = readIna226PowerW(currentPowerW);

  if (inaSuccess) {
    if (powerAccumValid) {
      unsigned long dtInaMicros = now - prevInaTimeMicros; // 符号なし減算でオーバーフロー補正
      float dtInaSec = (float)dtInaMicros / 1e6f;
      powerEnergyJAccum += 0.5f * (previousPowerW + currentPowerW) * dtInaSec;
      powerWindowDurationSec += dtInaSec;
    } else {
      powerAccumValid = true;
    }
    previousPowerW = currentPowerW;
    prevInaTimeMicros = now;
  } else {
    // 読み取り失敗時：無効測定値を積算せず、時刻を更新して失敗期間を積算から除外
    if (powerAccumValid) {
      prevInaTimeMicros = now;
    }
  }

  int raw2 = analogRead(analogPin2);
  int raw3 = analogRead(analogPin1); // 光センサ (GP28)

  // 光センサ値に対する一次遅れローパスフィルタ (LPF) 計算
  float dtSec = (float)dt / 1e6f;
  if (!lightFilterInitialized) {
    filteredLightRaw = (float)raw3;
    lightFilterInitialized = true;
  } else {
    if (lightTauSec > 0.0f) {
      float alpha = dtSec / (lightTauSec + dtSec);
      filteredLightRaw += alpha * ((float)raw3 - filteredLightRaw);
    } else {
      filteredLightRaw = (float)raw3;
    }
  }

  // リングバッファにデータを追加
  raw2Window[raw2Index] = raw2;
  raw2Index = (raw2Index + 1) % windowSize; // インデックスを循環させる

  // 下位10%と上位10%の値、およびその平均を計算
  auto [lowerValue, upperValue, trimmedMean] = calculateTrimmedMean(raw2Window, windowSize);

  // 正規化
  float flex = normalize((float)raw2 / 4095.0f, lowerValue, upperValue);
  float dflex = (flex - previousFlex)/dt; // 前回との差分
  previousFlex = flex; // 前回の値を更新

  // フィルタ後の光センサ値 (filteredLightRaw) に応じた振幅の連続線形減衰フィードバック
  float lightNorm = filteredLightRaw / 4095.0f; // 0.0 〜 1.0 に正規化
  float attenuationFactor = 1.0f - (lightFeedbackGain * lightNorm);
  if (attenuationFactor < minAmplitudeRatio) {
    attenuationFactor = minAmplitudeRatio;
  }
  float currentAmplitude = servoAmplitude * attenuationFactor;

  // サーボ制御 (絶対時刻 elapsed を使用した基準位相 omega*t + 摂動 phi)
  float psi = (float)elapsed / 1e6f * omega + phi;
  float zPrc = evaluatePRC(psi);

  // 摂動 phi の更新 (曲げセンサフィードバックと光センサフィードバックを同列に加算)
  float lightPhaseEffect = - lightFeedbackOmegaGain * omega * lightNorm;
  phi += (kappa_now * zPrc * flex + lightPhaseEffect) * ((float)dt / 1e6f);

  float currentCos = cosf(psi);
  myServo.write(servoCenter + currentAmplitude * currentCos);


  // データ保存は指定された間隔でのみ実行
  if (loopCounter % saveInterval == 0) {
    int raw1 = 0;
    if (powerAccumValid && powerWindowDurationSec > 0.0f) {
      float averagePowerW = powerEnergyJAccum / powerWindowDurationSec;
      raw1 = (int)(averagePowerW * POWER_W_TO_RAW + 0.5f);
      if (raw1 > POWER_RAW_MAX) {
        raw1 = POWER_RAW_MAX;
      }
      if (raw1 < 0) {
        raw1 = 0;
      }
      lastPowerRaw = raw1;

      // 保存区間のリセット（境界の最終測定値previousPowerW/prevInaTimeMicrosは次区間の初期値として引き継ぐ）
      powerEnergyJAccum = 0.0f;
      powerWindowDurationSec = 0.0f;
    } else {
      raw1 = lastPowerRaw;
    }

    // ログ用構造体
    CompressedLogData entry;
    uint32_t m24 = now >> 8;  // 24ビットに圧縮
    entry.micros24_0 = (uint8_t)(m24 & 0xFF);
    entry.micros24_1 = (uint8_t)((m24 >> 8) & 0xFF);
    entry.micros24_2 = (uint8_t)((m24 >> 16) & 0xFF);

    // analog0: phiを [0..2π) → 0..255 に圧縮
    float phiMod = fmodf((float)elapsed / 1e6f * omega + phi, 2.0f * (float)M_PI);
    if (phiMod < 0) phiMod += 2.0f * (float)M_PI;
    entry.analog0 = (uint8_t)(phiMod * (255.0f / (2.0f * (float)M_PI)));

    // analog1: 16mW単位へ四捨五入（最も近い16mW単位へ丸めて0..255に制限）
    int val16 = (raw1 + 8) >> 4;
    if (val16 > 255) {
      val16 = 255;
    }
    if (val16 < 0) {
      val16 = 0;
    }
    entry.analog1 = (uint8_t)val16;

    entry.analog2 = (uint8_t)(raw2 >> 4);

    entry.analog3 = (uint8_t)((int)filteredLightRaw >> 4);

    // バッファに書き込み
    if (logIndex < LOG_BUFFER_SIZE) {
      logBuffer[logIndex++] = entry;
    } else {
      // 1度だけWarnを出す
      if (!bufferOverflowed) {
        Serial.println("[WARN] log buffer overflow!");
        bufferOverflowed = true;
      }
    }

    // バッファ使用率 (10件毎に表示)
    if (logIndex % 10 == 0) {
      float usage = (float)logIndex / LOG_BUFFER_SIZE * 100.0f;
      //Serial.printf("[STATUS] buffer: %d/%d (%.1f%%)\n", logIndex, LOG_BUFFER_SIZE, usage);
    }
  }


  unsigned long now2 = micros();
  unsigned long dt2 = now2 - prevLoopEndTime2;
  // 周期制御
  if (dt2 < CONTROL_PERIOD_US) {
    delayMicroseconds(CONTROL_PERIOD_US - dt2);
    prevLoopEndTime2 = micros();
    //Serial.printf("[INFO] Loop took %lu us (expected %d us)\n", dt2, CONTROL_PERIOD_US);
  } else{
    prevLoopEndTime2 = micros();
    //Serial.printf("[WARN] Loop took too long: %lu us (expected %d us)\n", dt2, CONTROL_PERIOD_US);
  }

  // ループカウンタをインクリメント
  loopCounter++;
}

void setup() {
  pinMode(digitalInputPin, INPUT);
  Serial.begin(115200);
  analogReadResolution(12);

  Wire1.setSDA(inaSdaPin);
  Wire1.setSCL(inaSclPin);
  Wire1.begin();
  Wire1.setClock(400000);
  inaReady = detectIna226Address();
  if (inaReady) {
    initIna226();
  }

  myServo.attach(1);
  myServo.write(servoCenter); // 初期位置を中心に設定

  agent_id = readAgentIdFromFile(); // ユーザ実装の想定
  agentPort = 5000 + agent_id; // エージェント専用ポート

  // WiFi接続
  connectToWiFi(ssid, password);

  udp.begin(12345);

  // サーバー接続先を選択
  IPAddress serverIP1(192, 168, 13, 98);
  IPAddress serverIP2(192, 168, 13, 99);
  
  warmUpUDP(udp, serverIP1, serverPort);  // ServerUtils.cppの関数を呼び出し
  warmUpUDP(udp, serverIP2, serverPort);  
  
  while (true) {
    if (isServerReady(udp, serverIP1, serverPort)) {
      serverIP = serverIP1;
      Serial.println("[INFO] Connected to server at 192.168.13.98");
      break;
    } else if (isServerReady(udp, serverIP2, serverPort)) {
      serverIP = serverIP2;
      Serial.println("[INFO] Connected to server at 192.168.13.99");
      break;
    } else {
      Serial.println("[WARN] No servers are ready. Retrying in 1 second...");
      delay(1000);  // 1秒待機して再試行
    }
  }

  Serial.printf("Loaded agent_id: %d\n", agent_id);

  // 最初はメインポート（5000）でパラメータリクエスト
  requestParametersFromServer(udp, serverIP, serverPort, agent_id, readMonitorVoltageV(), omega, kappa, servoCenter, servoAmplitude, stopAgentId, stopDelaySeconds, feedbackTauSec, lightTauSec, lightFeedbackGain, lightFeedbackOmegaGain, minAmplitudeRatio, initialPhase, isRandomInitialPhase, prcHarmonics, prcCosCoeffs, prcSinCoeffs, PRC_MAX_HARMONICS);

  // パラメータ取得後、専用ポートに切り替え
  serverPort = agentPort;
  Serial.printf("[INFO] Switched to agent-specific port: %d\n", serverPort);

  // サーボモータを真ん中に動かす
  myServo.write(servoCenter); // パラメータ受信後の値で中心に設定
  Serial.println("[INFO] Servo moved to center position");

  Serial.println("[INFO] Ready to log in RAM");
  prevLoopEndTime = micros();
  prevLoopEndTime2 = prevLoopEndTime;

  // 初期状態をオフに設定
  paused = true;
  logIndex = 0;  // バッファインデックスを初期化
  resetPowerAccumulator();
  sendLogBuffer();
  kappa_now = kappa_init;
  srand(micros());
  Serial.println("[INFO] System is paused. Press the button to start.");
}

// UDPコマンド受信処理
void checkControlCommand() {
  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    char buf[16] = {0};
    udp.read(buf, sizeof(buf) - 1);
    if (strcmp(buf, "START") == 0 && paused == true) {
      paused = false;
      resetPowerAccumulator();
      startLoggingMillis = millis(); // ログ開始時刻を記録
      startLoggingMicros = micros(); // ログ開始時刻を記録
      lightFilterInitialized = false; // 光センサフィルタ初期化
      applyInitialPhase(); // 初期位相を設定
      Serial.println("[INFO] Received START command from server.");
      t_delay = (rand() / (float)RAND_MAX) * wait_max;
      startLoggingMicros += (unsigned long)(t_delay * 1e6f);
    } else if (strcmp(buf, "STOP") == 0 && paused == false) {
      paused = true;
      Serial.println("[INFO] Received STOP command from server.");
      sendLogBuffer();
      logIndex = 0;
      kappa_now = kappa_init;
    }
  }
}

void loop() {
  checkControlCommand();

  // サーバーからのパラメータに基づく停止処理
  if (!paused && stopAgentId == agent_id && stopAgentId != 0 && stopDelaySeconds > 0 && (millis() - startLoggingMillis >= (unsigned long)stopDelaySeconds * 1000)) {
    Serial.printf("[INFO] Agent %d stopping as per server: finalization phase (triggered after %d s).\\n", agent_id, stopDelaySeconds);
    
    // ファイナライズ期間は、ログ開始 (startLoggingMillis) から最大180秒後まで。
    unsigned long finalizationEndTime = startLoggingMillis + 180000UL; 

    // 現在時刻がファイナライズ終了時刻より前の場合のみ待機
    if (millis() < finalizationEndTime) {
        Serial.printf("[INFO] Entering finalization wait. Current: %lu, Target end: %lu (from startLoggingMillis: %lu)\\n", millis(), finalizationEndTime, startLoggingMillis);
        while(millis() < finalizationEndTime){
            delay(1000); 
        }
        Serial.println("[INFO] Finalization wait period finished.");
    } else {
        Serial.println("[INFO] Finalization period already passed or not applicable at stop trigger.");
    }

    paused = true;
    Serial.println("[INFO] Operation paused. Sending final log buffer.");
    sendLogBuffer();
    logIndex = 0;
    kappa_now = kappa_init; // kappaを初期値に戻す
    
    Serial.println("[INFO] Resetting stop parameters (stopAgentId, stopDelaySeconds) to prevent re-triggering.");
    stopAgentId = 0; 
    stopDelaySeconds = 0;
  }

  // kappaの更新ロジック (これは元のまま)
  if (!paused && (millis() - startLoggingMillis >= 5000)) {
    kappa_now = kappa;
  }

  bool currentButtonState = digitalRead(digitalInputPin);
  if (currentButtonState && !lastButtonState) {
    paused = !paused;
    Serial.println(paused ? "[INFO] Paused - Sending log from RAM" : "[INFO] Resumed");
    delay(300);  // チャタリング防止

    if (paused) {
      // ログ送信
      sendLogBuffer();
      // バッファ初期化
      logIndex = 0;
      kappa_now = kappa_init;

      // サーバーにパラメータをリクエスト（一時的にメインポートを使用）
      unsigned int tempPort = serverPort;
      serverPort = 5000; // メインポートに一時切り替え
      requestParametersFromServer(udp, serverIP, serverPort, agent_id, readMonitorVoltageV(), omega, kappa, servoCenter, servoAmplitude, stopAgentId, stopDelaySeconds, feedbackTauSec, lightTauSec, lightFeedbackGain, lightFeedbackOmegaGain, minAmplitudeRatio, initialPhase, isRandomInitialPhase, prcHarmonics, prcCosCoeffs, prcSinCoeffs, PRC_MAX_HARMONICS);
      serverPort = tempPort; // 専用ポートに戻す
      lastRequestTime = millis();  // リクエスト送信時刻を記録
    } else{
      resetPowerAccumulator();
      startLoggingMillis = millis(); // ログ開始時刻を記録
      startLoggingMicros = micros(); // ログ開始時刻を記録 
      lightFilterInitialized = false; // 光センサフィルタ初期化
      applyInitialPhase(); // 初期位相を設定
      t_delay = (rand() / (float)RAND_MAX) * wait_max;
      startLoggingMicros += (unsigned long)(t_delay * 1e6f);
    }
  }
  lastButtonState = currentButtonState;

  // ポーズ中に一定間隔でパラメータをリクエスト
  if (paused && millis() - lastRequestTime >= 10000) {
    while (WiFi.status() != WL_CONNECTED) {
      connectToWiFi(ssid, password);
    }
    Serial.println("[INFO] WiFi connected.");
    
    // パラメータリクエスト時は一時的にメインポートを使用
    unsigned int tempPort = serverPort;
    serverPort = 5000; // メインポートに一時切り替え
    requestParametersFromServer(udp, serverIP, serverPort, agent_id, readMonitorVoltageV(), omega, kappa, servoCenter, servoAmplitude, stopAgentId, stopDelaySeconds, feedbackTauSec, lightTauSec, lightFeedbackGain, lightFeedbackOmegaGain, minAmplitudeRatio, initialPhase, isRandomInitialPhase, prcHarmonics, prcCosCoeffs, prcSinCoeffs, PRC_MAX_HARMONICS);
    serverPort = tempPort; // 専用ポートに戻す
    lastRequestTime = millis();  // リクエスト送信時刻を更新
  }

  // 記録中
  if (!paused) {
    logSensorData();
  }
}
