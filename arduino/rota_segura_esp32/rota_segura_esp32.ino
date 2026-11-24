/*
  ROTA SEGURA - ESP32 (Bluetooth BLE embutido)

  O celular roda a IA (câmera -> CNN) e envia o risco por Bluetooth:
      'S' = seguro   'A' = atenção   'P' = perigo
  O ESP32 avisa na maquete com buzzer, LEDs e motor de vibração.
  O sensor HC-SR04 mede a distância real e serve de reforço: se o obstáculo estiver
  muito perto, o ESP32 avisa PERIGO mesmo que o celular não tenha detectado.
  A distância também é enviada ao app ("D:<cm>\n"), que mostra o valor na tela.

  Risco final = o MAIOR entre o risco da IA (celular) e o risco do sensor.
  Se o celular ficar 3 s sem enviar nada, o risco da IA volta a "seguro" (nunca
  fica preso em um alerta antigo) e o LED verde pisca para mostrar "sem celular".

  Bluetooth: nome "RotaSegura", serviço 0xFFE0, característica 0xFFE1 (o mesmo
  padrão do módulo HM-10, então o app funciona com as duas placas).

  Ligações: veja arduino/README.md
*/

#include <esp_arduino_version.h>
// O core do ESP32 mudou a biblioteca de Bluetooth na versão 4.0. Este código funciona nas duas:
#if ESP_ARDUINO_VERSION_MAJOR >= 4
#include <BLE.h>
#else
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLE2902.h>
#endif

// ---------------- pinos ----------------
const int PIN_BUZZER = 25;      // buzzer ATIVO (+ no pino, - no GND)
const int PIN_LED_VERDE = 14;
const int PIN_LED_AMARELO = 27;
const int PIN_LED_VERMELHO = 26;
const int PIN_MOTOR = 33;       // motor de vibração via transistor (ou módulo de vibração)
const int PIN_TRIG = 5;         // HC-SR04 TRIG
const int PIN_ECHO = 18;        // HC-SR04 ECHO (use divisor de tensão 5V -> 3,3V!)

// ---------------- ajustes ----------------
const int DIST_PERIGO_CM = 10;   // abaixo disso o sensor sozinho já indica PERIGO
const int DIST_ATENCAO_CM = 20;  // abaixo disso o sensor indica ATENÇÃO
const unsigned long TIMEOUT_APP_MS = 3000;
const unsigned long INTERVALO_SENSOR_MS = 120;

enum Risco { SEGURO = 0, ATENCAO = 1, PERIGO = 2 };

volatile int riscoApp = SEGURO;
volatile unsigned long ultimoComandoMs = 0;
int distanciaCm = -1;
unsigned long ultimoSensorMs = 0;
unsigned long ultimoEnvioMs = 0;
volatile bool celularConectado = false;

void recebeCodigo(char c) {
  if (c == 'S') riscoApp = SEGURO;
  else if (c == 'A') riscoApp = ATENCAO;
  else if (c == 'P') riscoApp = PERIGO;
  else return;
  ultimoComandoMs = millis();
}

// ---------------- Bluetooth (BLE) ----------------
#if ESP_ARDUINO_VERSION_MAJOR >= 4
BLECharacteristic caracteristica;

void iniciaBluetooth() {
  BLE.begin("RotaSegura");
  BLEServer servidor = BLE.createServer();
  servidor.onConnect([](BLEServer &, const BLEConnInfo &) { celularConectado = true; });
  servidor.onDisconnect([](BLEServer &, const BLEConnInfo &, uint8_t) {
    celularConectado = false;
    riscoApp = SEGURO;
  });
  servidor.advertiseOnDisconnect(true);  // volta a anunciar para reconectar
  BLEService servico = servidor.createService(BLEUUID((uint16_t)0xFFE0));
  caracteristica = servico.createCharacteristic(
      BLEUUID((uint16_t)0xFFE1),
      BLEProperty::Read | BLEProperty::Write | BLEProperty::WriteNR | BLEProperty::Notify,
      BLEPermission::ReadWriteOpen);
  caracteristica.onWrite([](BLECharacteristic &c, const BLEConnInfo &) {
    String valor = c.getStringValue();
    for (unsigned int i = 0; i < valor.length(); i++) recebeCodigo(valor[i]);
  });
  servidor.start();
  BLE.getAdvertising().addServiceUUID(BLEUUID((uint16_t)0xFFE0));
  BLE.startAdvertising();
}

void notifica(const char *texto, int n) {
  caracteristica.setValue((const uint8_t *)texto, n);
  caracteristica.notify();
}
#else
BLECharacteristic *caracteristica = nullptr;

class ServidorCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *) override { celularConectado = true; }
  void onDisconnect(BLEServer *servidor) override {
    celularConectado = false;
    riscoApp = SEGURO;
    servidor->getAdvertising()->start();  // volta a anunciar para reconectar
  }
};

class CaracteristicaCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) override {
    String valor = String(c->getValue().c_str());  // core 2.x (std::string) e 3.x (String)
    for (unsigned int i = 0; i < valor.length(); i++) recebeCodigo(valor[i]);
  }
};

void iniciaBluetooth() {
  BLEDevice::init("RotaSegura");
  BLEServer *servidor = BLEDevice::createServer();
  servidor->setCallbacks(new ServidorCallbacks());
  BLEService *servico = servidor->createService(BLEUUID((uint16_t)0xFFE0));
  caracteristica = servico->createCharacteristic(
      BLEUUID((uint16_t)0xFFE1),
      BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_WRITE |
          BLECharacteristic::PROPERTY_WRITE_NR | BLECharacteristic::PROPERTY_NOTIFY);
  caracteristica->addDescriptor(new BLE2902());
  caracteristica->setCallbacks(new CaracteristicaCallbacks());
  servico->start();
  BLEAdvertising *anuncio = BLEDevice::getAdvertising();
  anuncio->addServiceUUID(BLEUUID((uint16_t)0xFFE0));
  anuncio->setScanResponse(true);
  BLEDevice::startAdvertising();
}

void notifica(const char *texto, int n) {
  caracteristica->setValue((uint8_t *)texto, n);
  caracteristica->notify();
}
#endif

int mediDistanciaCm() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  unsigned long duracao = pulseIn(PIN_ECHO, HIGH, 25000UL);  // ~4 m no máximo; 0 = sem eco
  if (duracao == 0) return -1;
  return (int)(duracao / 58UL);
}

int riscoDoSensor(int cm) {
  if (cm < 0) return SEGURO;
  if (cm < DIST_PERIGO_CM) return PERIGO;
  if (cm < DIST_ATENCAO_CM) return ATENCAO;
  return SEGURO;
}

void enviaDistancia() {
  if (!celularConectado) return;
  char texto[12];
  int n = snprintf(texto, sizeof(texto), "D:%d\n", distanciaCm);
  notifica(texto, n);
}

void aplicaAvisos(int risco, unsigned long agora) {
  bool semCelular = (agora - ultimoComandoMs) > TIMEOUT_APP_MS;
  bool ligado = false;
  if (risco == PERIGO) ligado = (agora % 250) < 125;        // bipe rápido e contínuo
  else if (risco == ATENCAO) ligado = (agora % 900) < 120;  // bipe curto a cada ~1 s

  digitalWrite(PIN_BUZZER, ligado ? HIGH : LOW);
  digitalWrite(PIN_MOTOR, ligado ? HIGH : LOW);
  digitalWrite(PIN_LED_VERMELHO, risco == PERIGO ? HIGH : LOW);
  digitalWrite(PIN_LED_AMARELO, risco == ATENCAO ? HIGH : LOW);
  // Verde fixo = seguro com celular conectado; verde piscando = seguro, mas sem dados do celular.
  bool verde = (risco == SEGURO) && (!semCelular || (agora % 1000) < 100);
  digitalWrite(PIN_LED_VERDE, verde ? HIGH : LOW);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_LED_VERDE, OUTPUT);
  pinMode(PIN_LED_AMARELO, OUTPUT);
  pinMode(PIN_LED_VERMELHO, OUTPUT);
  pinMode(PIN_MOTOR, OUTPUT);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);

  iniciaBluetooth();
  Serial.println("Rota Segura pronto. Conecte pelo app (Arduino > Conectar).");
}

void loop() {
  unsigned long agora = millis();
  if (agora - ultimoSensorMs >= INTERVALO_SENSOR_MS) {
    ultimoSensorMs = agora;
    distanciaCm = mediDistanciaCm();
  }
  if (agora - ultimoEnvioMs >= 300) {
    ultimoEnvioMs = agora;
    enviaDistancia();
  }

  int ia = ((agora - ultimoComandoMs) > TIMEOUT_APP_MS) ? SEGURO : riscoApp;
  int sensor = riscoDoSensor(distanciaCm);
  int risco = max(ia, sensor);
  aplicaAvisos(risco, agora);
}
