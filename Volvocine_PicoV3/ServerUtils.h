#ifndef SERVER_UTILS_H
#define SERVER_UTILS_H

#include <WiFiUdp.h>

// サーバーの応答を確認する関数
bool isServerReady(WiFiUDP& udp, IPAddress serverIP, unsigned int serverPort);

// UDP通信のウォームアップ関数
void warmUpUDP(WiFiUDP& udp, IPAddress serverIP, unsigned int serverPort);

// ACKを待機する関数
bool waitForAck(WiFiUDP& udp, int agent_id, uint32_t expected_micros24, unsigned long timeout_ms = 1000);

#endif // SERVER_UTILS_H