import { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import type { LayersModel } from '@tensorflow/tfjs'
import { ArduinoLink, bluetoothSupported, RISK_CODE } from './arduino'
import { AlertTriangle, Bluetooth, BluetoothConnected, Camera, Check, CircleHelp, Download, Eye, ImagePlus, LoaderCircle, Radio, ShieldCheck, Volume2, VolumeX, Vibrate, X } from 'lucide-react'

const CLASSES = ['sem_obstaculo', 'com_obstaculo'] as const  // mesma ordem de ml/comum.py
const IMAGE_SIZE = 160 // mesmo tamanho usado no treino (ml/comum.py)
type ModelInfo = { acuracia_teste?: number; acuracia_balanceada?: number; revocacao_obstaculo?: number; amostras_teste?: number; limiar_obstaculo?: number }
type Risk = (typeof CLASSES)[number]
type Mode = 'monitor' | 'collect'

const LABELS: Record<Risk, string> = { sem_obstaculo: 'Caminho livre', com_obstaculo: 'Obstáculo à frente' }
const RISK_COPY: Record<Risk, string> = {
  sem_obstaculo: 'Nenhum objeto no caminho à frente.',
  com_obstaculo: 'Há um objeto no caminho. Pare e confirme com seus recursos habituais.',
}

function makeSessionId() {
  return Math.random().toString(36).slice(2, 8)
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const modelRef = useRef<LayersModel | null>(null)
  const previousRisk = useRef<Risk | null>(null)
  const lastAlertAt = useRef(0)
  const [cameraOn, setCameraOn] = useState(false)
  const [mode, setMode] = useState<Mode>('monitor')
  const [risk, setRisk] = useState<Risk | null>(null)
  const [confidence, setConfidence] = useState(0)
  const [modelState, setModelState] = useState<'loading' | 'ready' | 'missing'>('loading')
  const [cameraMessage, setCameraMessage] = useState('A câmera ainda não foi iniciada.')
  const [alertMessage, setAlertMessage] = useState('Alertas habilitados após iniciar a câmera.')
  const [captureLabel, setCaptureLabel] = useState<Risk>('com_obstaculo')
  const [sessionId] = useState(makeSessionId)
  const [soundOn, setSoundOn] = useState(true)
  const [vibrationOn, setVibrationOn] = useState(true)
  const [busy, setBusy] = useState(false)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const linkRef = useRef<ArduinoLink | null>(null)
  const [arduino, setArduino] = useState<'off' | 'connecting' | 'on'>('off')
  const [distance, setDistance] = useState<number | null>(null)
  const [arduinoMessage, setArduinoMessage] = useState(
    bluetoothSupported() ? 'Liga o buzzer, LED e vibração da maquete por Bluetooth.' : 'Bluetooth indisponível neste navegador (use o Chrome no Android ou o Bluefy no iPhone).')
  const dangerThreshold = useRef<number | null>(null)

  useEffect(() => {
    let active = true
    tf.loadLayersModel('/model/model.json')
      .then((model) => {
        if (!active) { model.dispose(); return }
        modelRef.current = model
        setModelState('ready')
      })
      .catch(() => { if (active) setModelState('missing') })
    fetch('/model/info.json')
      .then((response) => (response.ok ? response.json() : null))
      .then((info: ModelInfo | null) => {
        if (active && info) { setModelInfo(info); dangerThreshold.current = info.limiar_obstaculo ?? null }
      })
      .catch(() => undefined)
    return () => {
      active = false
      modelRef.current?.dispose()
      modelRef.current = null
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  useEffect(() => {
    if (!cameraOn || mode !== 'monitor' || modelState !== 'ready') return
    const timer = window.setInterval(() => { void classifyFrame() }, 500)
    return () => window.clearInterval(timer)
  }, [cameraOn, mode, modelState])

  async function startCamera() {
    setCameraMessage('Solicitando acesso à câmera...')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setCameraOn(true)
      setCameraMessage('Câmera ativa. Mantenha o celular apontado para o percurso.')
      setAlertMessage('Alertas por som e vibração prontos.')
      previousRisk.current = null
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Permissão não concedida.'
      setCameraMessage(`Não foi possível abrir a câmera: ${message}`)
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraOn(false)
    setCameraMessage('Câmera desligada.')
  }

  async function classifyFrame() {
    const video = videoRef.current
    const model = modelRef.current
    if (!video || !model || video.readyState < 2) return
    // Recorta o quadrado central do vídeo (como no treino) e reduz para 160x160.
    const input = tf.tidy(() => {
      const frame = tf.browser.fromPixels(video)
      const [height, width] = frame.shape
      const side = Math.min(height, width)
      const top = Math.floor((height - side) / 2)
      const left = Math.floor((width - side) / 2)
      return frame
        .slice([top, left, 0], [side, side, 3])
        .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
        .toFloat()
        .div(255)
        .expandDims(0)
    })
    try {
      const prediction = model.predict(input) as tf.Tensor
      const values = await prediction.data()
      prediction.dispose()
      // Regra de decisão da avaliação (ml/6_avaliacao.py): "perigo" quando P(perigo) passa do limiar
      // escolhido na validação; senão, a classe mais provável entre seguro e atenção.
      const obstacleIndex = CLASSES.indexOf('com_obstaculo')
      const threshold = dangerThreshold.current ?? 0.5
      const maxIndex = values[obstacleIndex] >= threshold ? obstacleIndex : 1 - obstacleIndex
      const nextRisk = CLASSES[maxIndex]
      setRisk(nextRisk)
      setConfidence(values[maxIndex] ?? 0)
      void linkRef.current?.send(RISK_CODE[nextRisk])  // o Arduino apita/acende conforme o risco
      if (previousRisk.current !== nextRisk || (nextRisk === 'com_obstaculo' && Date.now() - lastAlertAt.current > 3000)) {
        announceRisk(nextRisk)
      }
      previousRisk.current = nextRisk
    } catch {
      setAlertMessage('Falha ao analisar o quadro. Verifique o modelo exportado.')
    } finally {
      input.dispose()
    }
  }

  async function connectArduino() {
    if (arduino === 'on') { linkRef.current?.disconnect(); return }
    setArduino('connecting')
    setArduinoMessage('Escolha o dispositivo "RotaSegura" (ESP32) ou "HMSoft" (HM-10) na lista.')
    const link = new ArduinoLink(
      (cm) => setDistance(cm),
      () => { linkRef.current = null; setArduino('off'); setDistance(null); setArduinoMessage('Arduino desconectado.') },
    )
    try {
      await link.connect()
      linkRef.current = link
      setArduino('on')
      setArduinoMessage(`Conectado a ${link.name}. O Arduino recebe o risco a cada análise.`)
    } catch (error) {
      setArduino('off')
      const message = error instanceof Error ? error.message : ''
      setArduinoMessage(message.includes('cancel') ? 'Conexão cancelada.' : `Não foi possível conectar: ${message || 'verifique se o Bluetooth está ligado.'}`)
    }
  }

  function announceRisk(nextRisk: Risk) {
    if (nextRisk === 'sem_obstaculo') {
      setAlertMessage('Nenhum alerta de risco alto neste momento.')
      return
    }
    lastAlertAt.current = Date.now()
    setAlertMessage(RISK_COPY[nextRisk])
    if (vibrationOn && 'vibrate' in navigator) navigator.vibrate([250, 120, 350])
    if (soundOn) {
      const audio = new AudioContext()
      const oscillator = audio.createOscillator()
      const gain = audio.createGain()
      oscillator.type = 'sine'
      oscillator.frequency.value = 880
      gain.gain.value = 0.12
      oscillator.connect(gain)
      gain.connect(audio.destination)
      oscillator.start()
      oscillator.stop(audio.currentTime + (0.42))
      oscillator.onended = () => { void audio.close() }
    }
  }

  function captureImage() {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || !video.videoWidth) return
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const context = canvas.getContext('2d')
    if (!context) return
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => {
      if (!blob) return
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `${captureLabel}_${sessionId}_${Date.now()}.jpg`
      anchor.click()
      URL.revokeObjectURL(url)
      setAlertMessage(`Imagem salva com rótulo “${LABELS[captureLabel]}”. Mova-a para data/maquete/${captureLabel}/ e rode o pipeline de novo.`)
    }, 'image/jpeg', 0.92)
  }

  async function handleCapture() {
    if (busy) return
    setBusy(true)
    captureImage()
    window.setTimeout(() => setBusy(false), 350)
  }

  const riskClass = risk === 'com_obstaculo' ? 'perigo' : risk === 'sem_obstaculo' ? 'seguro' : 'unknown'

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#inicio" aria-label="Rota Segura, início">
          <span className="brand-mark"><Eye size={21} strokeWidth={2.2} /></span>
          <span><strong>ROTA SEGURA</strong><small>LABORATÓRIO DE ACESSIBILIDADE</small></span>
        </a>
        <div className="topbar-right"><span className="prototype-tag"><span /> PROTÓTIPO ACADÊMICO</span><button className="icon-button help-button" title="Sobre o protótipo" aria-label="Sobre o protótipo" onClick={() => document.getElementById('limites')?.scrollIntoView({ behavior: 'smooth' })}><CircleHelp size={19} /></button></div>
      </header>

      <section className="intro" id="inicio">
        <div className="intro-copy"><p className="eyebrow">VISÃO COMPUTACIONAL · ACESSIBILIDADE</p><h1>Um caminho mais <em>perceptível.</em></h1><p>O celular analisa a imagem da câmera e sinaliza o nível de risco. Um protótipo para testar, medir e melhorar com dados próprios.</p></div>
        <div className="model-status"><span className={`status-led ${modelState}`} /> <div><small>MODELO DE IA</small><strong>{modelState === 'loading' ? 'Carregando…' : modelState === 'ready' ? (modelInfo?.acuracia_teste ? `Treinado · ${Math.round(modelInfo.acuracia_teste * 100)}% no teste` : 'Treinado e carregado') : 'Aguardando treinamento'}</strong></div></div>
      </section>

      <section className="workbench" aria-label="Painel da câmera">
        <div className="camera-column">
          <div className="camera-toolbar"><div className="section-label"><Radio size={16} /><span>LEITURA AO VIVO</span></div><div className="camera-actions">{cameraOn ? <button className="button button-subtle" onClick={stopCamera}><X size={16} /> Encerrar</button> : <button className="button button-primary" onClick={() => void startCamera()}><Camera size={16} /> Abrir câmera</button>}</div></div>
          <div className={`camera-frame ${cameraOn ? 'is-live' : ''}`}>
            <video ref={videoRef} className="camera-video" playsInline muted aria-label="Imagem capturada pela câmera" />
            {!cameraOn && <div className="camera-placeholder"><div className="camera-glyph"><Camera size={26} /></div><strong>O percurso aparece aqui</strong><span>Use a câmera traseira para testar a maquete.</span></div>}
            {cameraOn && <><div className="analysis-zone" aria-hidden="true"><span>ÁREA ANALISADA</span></div><div className="frame-corner corner-tl" /><div className="frame-corner corner-tr" /><div className="frame-corner corner-bl" /><div className="frame-corner corner-br" /><span className="live-indicator"><span /> AO VIVO</span></>}
            <canvas ref={canvasRef} className="hidden-canvas" />
          </div>
          <div className="camera-foot"><span className="camera-message"><span className={`small-led ${cameraOn ? 'on' : ''}`} />{cameraMessage}</span><span className="frame-rate">{cameraOn && mode === 'monitor' && modelState === 'ready' ? '2 análises / s' : '—'}</span></div>
        </div>

        <aside className="results-column">
          <div className="mode-switch" role="tablist" aria-label="Modo de uso">
            <button role="tab" aria-selected={mode === 'monitor'} className={mode === 'monitor' ? 'selected' : ''} onClick={() => setMode('monitor')}><Eye size={16} /> Monitorar</button>
            <button role="tab" aria-selected={mode === 'collect'} className={mode === 'collect' ? 'selected' : ''} onClick={() => setMode('collect')}><ImagePlus size={16} /> Coletar dados</button>
          </div>

          {mode === 'monitor' ? <>
            <div className={`risk-panel risk-${riskClass}`} aria-live="polite">
              <div className="risk-heading"><span className="risk-icon">{risk === 'sem_obstaculo' ? <ShieldCheck size={23} /> : risk ? <AlertTriangle size={23} /> : <Eye size={22} />}</span><span className="risk-caption">CLASSIFICAÇÃO ATUAL</span></div>
              <strong className="risk-title">{risk ? LABELS[risk] : modelState === 'ready' ? 'Aguardando imagem' : 'Modelo não treinado'}</strong>
              <p>{risk ? RISK_COPY[risk] : modelState === 'ready' ? 'Inicie a câmera para analisar o percurso.' : 'Rode bash ml/executar_tudo.sh para treinar e exportar o modelo.'}</p>
              {risk && <div className="confidence"><div><span>Confiança do modelo</span><strong>{Math.round(confidence * 100)}%</strong></div><div className="confidence-track"><span style={{ width: `${Math.round(confidence * 100)}%` }} /></div></div>}
            </div>
            <div className="alerts-box"><div className="box-heading"><span>ALERTAS DO DISPOSITIVO</span><span className="ready-label"><Check size={13} /> PRONTOS</span></div><button className="setting-row" aria-pressed={soundOn} onClick={() => setSoundOn(!soundOn)}><span className="setting-name">{soundOn ? <Volume2 size={17} /> : <VolumeX size={17} />} Som</span><span className={`toggle ${soundOn ? 'enabled' : ''}`}><i /></span></button><button className="setting-row" aria-pressed={vibrationOn} onClick={() => setVibrationOn(!vibrationOn)}><span className="setting-name"><Vibrate size={17} /> Vibração</span><span className={`toggle ${vibrationOn ? 'enabled' : ''}`}><i /></span></button><button className="setting-row" disabled={!bluetoothSupported() || arduino === 'connecting'} aria-pressed={arduino === 'on'} onClick={() => void connectArduino()}><span className="setting-name">{arduino === 'on' ? <BluetoothConnected size={17} /> : <Bluetooth size={17} />} Arduino {arduino === 'on' && distance !== null ? `· sensor ${distance} cm` : ''}</span><span className="arduino-action">{arduino === 'on' ? 'Desconectar' : arduino === 'connecting' ? 'Conectando…' : 'Conectar'}</span></button><p className="arduino-note">{arduinoMessage}</p></div>
            <p className="alert-status" aria-live="polite">{alertMessage}</p>
          </> : <>
            <div className="collect-panel"><div className="collect-title"><span className="collect-icon"><ImagePlus size={20} /></span><div><h2>Nova imagem</h2><p>Fotos da maquete reforçam a base de treino.</p></div></div>
              <label className="field-label" htmlFor="capture-label">Rótulo da situação</label><select id="capture-label" value={captureLabel} onChange={(event) => setCaptureLabel(event.target.value as Risk)}>{CLASSES.map((item) => <option value={item} key={item}>{LABELS[item]}</option>)}</select>
              <button className="button button-capture" disabled={!cameraOn || busy} onClick={() => void handleCapture()}><Download size={17} /> Baixar imagem rotulada</button>
              <div className="session-note"><span className="session-dot" /><span>Sessão <strong>{sessionId}</strong><br />Mantenha este código para agrupar imagens do mesmo cenário.</span></div>
              <p className="collect-tip">Colete várias distâncias, objetos e iluminações por classe. Não use quadros consecutivos quase idênticos.</p>
            </div>
          </>}
        </aside>
      </section>

      {modelState === 'missing' && <section className="model-notice"><div className="notice-mark"><LoaderCircle size={18} /></div><p><strong>O modelo ainda não está no app.</strong> Rode <code>bash ml/executar_tudo.sh</code> (ou as etapas 1 a 6) para coletar dados, treinar e exportar o modelo para <code>public/model</code>.</p><span>ML / ARTIFACTS → PUBLIC / MODEL</span></section>}

      <section className="workflow-strip"><div className="workflow-heading"><p className="eyebrow">DO DADO AO ALERTA</p><h2>O processo do projeto</h2></div><div className="workflow-steps"><div><span>01</span><strong>Coletar</strong><small>Fotos da maquete (base oficial)</small></div><div><span>02</span><strong>Rotular</strong><small>Detector pronto marca os obstáculos</small></div><div><span>03</span><strong>Treinar</strong><small>CNN própria, criada do zero</small></div><div><span>04</span><strong>Avaliar</strong><small>{modelInfo?.acuracia_teste ? `${Math.round(modelInfo.acuracia_teste * 100)}% de acerto na maquete · ${Math.round((modelInfo.revocacao_obstaculo ?? 0) * 100)}% dos obstáculos detectados` : 'Teste e matriz de confusão'}</small></div></div></section>

      <footer className="disclaimer" id="limites"><span className="disclaimer-mark"><AlertTriangle size={16} /></span><p><strong>Protótipo para demonstração acadêmica.</strong> Não é um dispositivo de navegação seguro nem substitui bengala, cão-guia ou orientação profissional. A câmera pode falhar, e a classificação não mede distância real. Teste somente em ambiente controlado, com acompanhamento.</p><span className="course-code">2026.2 · NSIN-0104</span></footer>
    </main>
  )
}