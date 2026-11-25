// Conexão Bluetooth (BLE) entre o app e o Arduino/ESP32 da maquete.
//
// Protocolo (texto simples, igual nos dois sketches em arduino/):
//   app -> Arduino: um caractere com o risco classificado pela IA: 'S' (seguro), 'A' (atenção), 'P' (perigo).
//                   É reenviado a cada análise (2x/s); se o Arduino ficar 3 s sem receber, ele volta a "seguro".
//   Arduino -> app: "D:<cm>\n" com a distância medida pelo HC-SR04 (ou "D:-1" sem leitura).
//
// Usa o serviço BLE 0xFFE0 / característica 0xFFE1, o padrão do módulo HM-10; o sketch do ESP32 usa os mesmos.
// Web Bluetooth funciona no Chrome do Android e do computador; no iPhone só pelo navegador Bluefy.

const SERVICE = 0xffe0
const CHARACTERISTIC = 0xffe1

type Characteristic = {
  writeValueWithoutResponse?: (data: BufferSource) => Promise<void>
  writeValue: (data: BufferSource) => Promise<void>
  startNotifications: () => Promise<unknown>
  addEventListener: (type: string, listener: (event: Event) => void) => void
  value?: DataView
}
type Device = {
  name?: string
  gatt?: { connect: () => Promise<{ getPrimaryService: (s: number) => Promise<{ getCharacteristic: (c: number) => Promise<Characteristic> }> }>; disconnect: () => void; connected: boolean }
  addEventListener: (type: string, listener: () => void) => void
}
type BluetoothApi = { requestDevice: (options: object) => Promise<Device> }

// Com obstáculo -> 'P' (o Arduino apita/vibra como perigo); sem obstáculo -> 'S'.
export const RISK_CODE = { sem_obstaculo: 'S', com_obstaculo: 'P' } as const

export function bluetoothSupported(): boolean {
  return typeof navigator !== 'undefined' && 'bluetooth' in navigator
}

export class ArduinoLink {
  private device: Device | null = null
  private characteristic: Characteristic | null = null
  private writing = false
  private buffer = ''
  private encoder = new TextEncoder()
  private decoder = new TextDecoder()

  constructor(
    private onDistance: (cm: number | null) => void,
    private onDisconnect: () => void,
  ) {}

  get name(): string {
    return this.device?.name ?? 'Arduino'
  }

  async connect(): Promise<void> {
    const bluetooth = (navigator as unknown as { bluetooth: BluetoothApi }).bluetooth
    this.device = await bluetooth.requestDevice({
      filters: [{ services: [SERVICE] }, { namePrefix: 'RotaSegura' }, { namePrefix: 'HMSoft' }],
      optionalServices: [SERVICE],
    })
    this.device.addEventListener('gattserverdisconnected', () => {
      this.characteristic = null
      this.onDisconnect()
    })
    const server = await this.device.gatt!.connect()
    const service = await server.getPrimaryService(SERVICE)
    this.characteristic = await service.getCharacteristic(CHARACTERISTIC)
    this.characteristic.addEventListener('characteristicvaluechanged', (event) => {
      const value = (event.target as unknown as Characteristic).value
      if (value) this.receive(this.decoder.decode(value))
    })
    await this.characteristic.startNotifications()
  }

  disconnect(): void {
    this.device?.gatt?.disconnect()
  }

  // Envia o risco. Se uma escrita ainda estiver em andamento, esta é pulada (a próxima análise reenvia).
  async send(code: string): Promise<void> {
    const c = this.characteristic
    if (!c || this.writing) return
    this.writing = true
    try {
      const data = this.encoder.encode(code)
      if (c.writeValueWithoutResponse) await c.writeValueWithoutResponse(data)
      else await c.writeValue(data)
    } catch {
      // Falha pontual de rádio: ignora, o próximo envio corrige.
    } finally {
      this.writing = false
    }
  }

  private receive(text: string): void {
    // O HM-10 pode quebrar as mensagens em pedaços: junta até achar o fim de linha.
    this.buffer = (this.buffer + text).slice(-64)
    let newline = this.buffer.indexOf('\n')
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline).trim()
      this.buffer = this.buffer.slice(newline + 1)
      const match = /^D:(-?\d+)/.exec(line)
      if (match) {
        const cm = Number(match[1])
        this.onDistance(cm >= 0 ? cm : null)
      }
      newline = this.buffer.indexOf('\n')
    }
  }
}
