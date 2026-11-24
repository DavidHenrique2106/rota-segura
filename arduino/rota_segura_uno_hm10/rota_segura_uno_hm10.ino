/*
  ROTA SEGURA - Arduino Uno/Nano + módulo Bluetooth BLE HM-10

  Mesma lógica da versão ESP32:
    - recebe do celular o risco classificado pela IA: 'S' seguro, 'A' atenção, 'P' perigo;
    - mede a distância com o HC-SR04 e usa como reforço (muito perto = PERIGO);
    - risco final = o MAIOR dos dois; avisa com buzzer, LEDs e motor de vibração;
    - envia a distância ao app ("D:<cm>\n");
    - sem dados do celular por 3 s, o risco da IA volta a "seguro".

  IMPORTANTE: use o HM-10 (BLE). O HC-05/HC-06 é Bluetooth "clássico" e o navegador
  do celular não consegue se conectar a ele.

  Ligações: veja arduino/README.md
*/

#include <SoftwareSerial.h>

// ---------------- pinos ----------------
const int PIN_BLE_RX = 2;       // liga no TX do HM-10
const int PIN_BLE_TX = 3;       // liga no RX do HM-10 (com divisor de tensão 5V -> 3,3V)
const int PIN_TRIG = 4;         // HC-SR04 TRIG
const int PIN_ECHO = 5;         // HC-SR04 ECHO
const int PIN_MOTOR = 6;        // motor de vibração via transistor (ou módulo de vibração)
const int PIN_BUZZER = 8;       // buzzer ATIVO
const int PIN_LED_VERDE = 9;
const int PIN_LED_AMARELO = 10;
const int PIN_LED_VERMELHO = 11;

// ---------------- ajustes ----------------
const int DIST_PERIGO_CM = 10;
const int DIST_ATENCAO_CM = 20;
const unsigned long TIMEOUT_APP_MS = 3000;
const unsigned long INTERVALO_SENSOR_MS = 120;

enum Risco { SEGURO = 0, ATENCAO = 1, PERIGO = 2 };

SoftwareSerial ble(PIN_BLE_RX, PIN_BLE_TX);
int riscoApp = SEGURO;
unsigned long ultimoComandoMs = 0;
int distanciaCm = -1;
unsigned long ultimoSensorMs = 0;
unsigned long ultimoEnvioMs = 0;

int mediDistanciaCm() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  unsigned long duracao = pulseIn(PIN_ECHO, HIGH, 25000UL);
  if (duracao == 0) return -1;
  return (int)(duracao / 58UL);
}

int riscoDoSensor(int cm) {
  if (cm < 0) return SEGURO;
  if (cm < DIST_PERIGO_CM) return PERIGO;
  if (cm < DIST_ATENCAO_CM) return ATENCAO;
  return SEGURO;
}

void leCelular() {
  while (ble.available()) {
    char c = (char)ble.read();
    if (c == 'S') riscoApp = SEGURO;
    else if (c == 'A') riscoApp = ATENCAO;
    else if (c == 'P') riscoApp = PERIGO;
    else continue;
    ultimoComandoMs = millis();
  }
}

void aplicaAvisos(int risco, unsigned long agora) {
  bool semCelular = (agora - ultimoComandoMs) > TIMEOUT_APP_MS;
  bool ligado = false;
  if (risco == PERIGO) ligado = (agora % 250) < 125;
  else if (risco == ATENCAO) ligado = (agora % 900) < 120;

  digitalWrite(PIN_BUZZER, ligado ? HIGH : LOW);
  digitalWrite(PIN_MOTOR, ligado ? HIGH : LOW);
  digitalWrite(PIN_LED_VERMELHO, risco == PERIGO ? HIGH : LOW);
  digitalWrite(PIN_LED_AMARELO, risco == ATENCAO ? HIGH : LOW);
  bool verde = (risco == SEGURO) && (!semCelular || (agora % 1000) < 100);
  digitalWrite(PIN_LED_VERDE, verde ? HIGH : LOW);
}

void setup() {
  Serial.begin(9600);
  ble.begin(9600);  // velocidade padrão do HM-10
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_LED_VERDE, OUTPUT);
  pinMode(PIN_LED_AMARELO, OUTPUT);
  pinMode(PIN_LED_VERMELHO, OUTPUT);
  pinMode(PIN_MOTOR, OUTPUT);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  Serial.println(F("Rota Segura pronto. Conecte pelo app (Arduino > Conectar)."));
}

void loop() {
  unsigned long agora = millis();
  leCelular();
  if (agora - ultimoSensorMs >= INTERVALO_SENSOR_MS) {
    ultimoSensorMs = agora;
    distanciaCm = mediDistanciaCm();
  }
  if (agora - ultimoEnvioMs >= 300) {
    ultimoEnvioMs = agora;
    ble.print(F("D:"));
    ble.print(distanciaCm);
    ble.print('\n');
  }
  int ia = ((agora - ultimoComandoMs) > TIMEOUT_APP_MS) ? SEGURO : riscoApp;
  int risco = max(ia, riscoDoSensor(distanciaCm));
  aplicaAvisos(risco, agora);
}
