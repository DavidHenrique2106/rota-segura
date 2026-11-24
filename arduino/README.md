# Arduino da maquete

O celular faz a parte de IA: a câmera vê o caminho e a CNN classifica o risco. O resultado vai por **Bluetooth (BLE)** para a placa, que avisa na maquete com **buzzer, LEDs e vibração**. O sensor ultrassônico **HC-SR04** mede a distância real e funciona como reforço.

```
câmera do celular ──► IA (CNN no navegador) ──► 'S' / 'A' / 'P' por Bluetooth ──► Arduino: buzzer + LEDs + vibração
                                                         ◄── "D:23" (distância do HC-SR04) ◄──
```

**Risco final no Arduino = o maior entre o risco da IA e o risco do sensor.**

| Situação | O que acontece |
|---|---|
| IA diz **perigo**, ou o sensor mede menos de **10 cm** | LED vermelho, bipe rápido e vibração |
| IA diz **atenção**, ou o sensor mede **10–20 cm** | LED amarelo, bipe curto a cada ~1 s |
| Seguro, com o celular conectado | LED verde aceso |
| Seguro, mas sem dados do celular há 3 s | LED verde piscando (só o sensor está valendo) |

As distâncias ficam em `DIST_PERIGO_CM` e `DIST_ATENCAO_CM`, no começo do código. Ajustem para a escala da maquete.

---

## Qual placa usar

| Opção | Código | Observação |
|---|---|---|
| **ESP32** (recomendado) | `rota_segura_esp32/` | Já tem Bluetooth BLE; aparece no celular como **RotaSegura** |
| **Arduino Uno/Nano + HM-10** | `rota_segura_uno_hm10/` | O HM-10 aparece como **HMSoft** |

O **HC-05/HC-06 não funciona**: é Bluetooth "clássico", e o navegador do celular só conversa com BLE.

---

## Ligações: ESP32

| Componente | Pino do componente | ESP32 |
|---|---|---|
| Buzzer **ativo** | + / − | GPIO 25 / GND |
| LED verde (com resistor de 220 Ω) | + | GPIO 14 |
| LED amarelo (220 Ω) | + | GPIO 27 |
| LED vermelho (220 Ω) | + | GPIO 26 |
| Motor de vibração (via transistor NPN 2N2222 + diodo, ou módulo de vibração) | sinal | GPIO 33 |
| HC-SR04 | VCC / GND | 5V (VIN) / GND |
| HC-SR04 | TRIG | GPIO 5 |
| HC-SR04 | ECHO | GPIO 18, **com divisor de tensão** (1 kΩ + 2 kΩ), porque o ECHO sai em 5 V e o ESP32 aceita 3,3 V |

Na Arduino IDE: instale a placa "esp32 by Espressif" no Gerenciador de Placas, escolha **ESP32 Dev Module** e faça o upload.

## Ligações: Arduino Uno/Nano + HM-10

| Componente | Pino do componente | Arduino |
|---|---|---|
| HM-10 | VCC / GND | 5V / GND (a maioria das plaquinhas HM-10 aceita 5 V; confira a sua) |
| HM-10 | TX | D2 |
| HM-10 | RX | D3, **com divisor de tensão** (1 kΩ + 2 kΩ) |
| HC-SR04 | VCC / GND / TRIG / ECHO | 5V / GND / D4 / D5 |
| Motor de vibração (transistor ou módulo) | sinal | D6 |
| Buzzer **ativo** | + / − | D8 / GND |
| LED verde / amarelo / vermelho (220 Ω cada) | + | D9 / D10 / D11 |

O motorzinho de vibração **não pode ser ligado direto no pino**, porque puxa corrente demais. Use um módulo de vibração pronto, ou um transistor NPN (base no pino por um resistor de 1 kΩ) com um diodo em paralelo com o motor.

---

## Como usar na apresentação

1. Ligue a placa (USB ou bateria).
2. No celular **Android**, abra https://rota-segura-sepia.vercel.app no **Chrome** e ligue o Bluetooth e a localização do celular.
   - No **iPhone**, use o navegador **Bluefy** (App Store). O Safari não tem Bluetooth para sites.
3. Toque em **Abrir câmera**. Depois, em "Alertas do dispositivo", toque em **Arduino · Conectar** e escolha **RotaSegura** (ESP32) ou **HMSoft** (HM-10).
4. Aponte o celular para o percurso. O Arduino apita conforme o risco, e a distância medida pelo sensor aparece no app.

## Protocolo (para quem quiser mexer)

- Bluetooth: serviço `0xFFE0`, característica `0xFFE1` (padrão do HM-10; o ESP32 usa o mesmo).
- **Celular → placa:** um caractere a cada análise (2 vezes por segundo): `S`, `A` ou `P`.
- **Placa → celular:** `D:<cm>\n` a cada 300 ms (`D:-1` quando o sensor não tem leitura).
