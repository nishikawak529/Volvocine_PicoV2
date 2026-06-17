#include "ServerUtils.h"
#include <Arduino.h>

bool isServerReady(WiFiUDP& udp, IPAddress serverIP, unsigned int serverPort) {
  const char* handshakeMessage = "HELLO";
  const int timeoutMs = 1000;  // 応答待ちタイムアウト (ミリ秒)
  char response[10];

  // ハンドシェイクメッセージを送信
  udp.beginPacket(serverIP, serverPort);
  udp.write(handshakeMessage);
  udp.endPacket();

  // 応答を待つ
  unsigned long startTime = millis();
  while (millis() - startTime < timeoutMs) {
    int packetSize = udp.parsePacket();
    if (packetSize > 0) {
      int len = udp.read(response, sizeof(response) - 1);
      if (len > 0) {
        response[len] = '\0';  // 文字列終端を追加
        if (strcmp(response, "READY") == 0) {
          Serial.println("[INFO] Server is ready.");
          return true;
        }
      }
    }
  }

  Serial.println("[WARN] No response from server.");
  return false;
}

void warmUpUDP(WiFiUDP& udp, IPAddress serverIP, unsigned int serverPort) {
  udp.beginPacket(serverIP, serverPort);
  udp.write((uint8_t)0);  // ダミーデータ送信
  udp.endPacket();
  delay(50);  // 少し待機
}

bool waitForAck(WiFiUDP& udp, int agent_id, uint32_t expected_micros24, unsigned long timeout_ms) {
  unsigned long start = millis();
  while (millis() - start < timeout_ms) {
    int len = udp.parsePacket();
    if (len >= 4) {
      uint8_t buffer[6] = {0};  // 念のため初期化
      udp.read(buffer, len);
      if ((uint8_t)buffer[0] != agent_id) continue;
      uint32_t receivedMicros24 = buffer[1] | (buffer[2] << 8) | (buffer[3] << 16);
      if (receivedMicros24 == expected_micros24) {
        Serial.println("[INFO] ACK received.");
        return true;
      } else {
        Serial.printf("[WARN] ACK mismatch: expected %lu, got %lu\n", expected_micros24, receivedMicros24);
      }
    }
    delay(10); // 少し待機して再チェック
  }
  return false; // タイムアウト
}