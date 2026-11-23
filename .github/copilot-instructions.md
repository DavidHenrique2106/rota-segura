# Projeto Rota Segura

- Pipeline de IA em `ml/`, um script por etapa: 1 coleta, 2 rotulagem automática das fotos reais (OWLv2, só como rotulador), 3 divisão treino/validação/teste, 4 montagem da base rotulada (fotos reais + montagens), 5 treinamento, 6 avaliação, 7 exportação para o app. Configuração comum em `ml/comum.py`.
- O modelo do app é uma CNN treinada do zero (TensorFlow/Keras 2 via `tf_keras`). Modelos pré-treinados só podem ser usados para rotular dados na etapa 2, nunca no app.
- Classes e ordem de saída devem permanecer `seguro`, `atencao`, `perigo` em `ml/comum.py` e `src/App.tsx`; entrada 160x160 (recorte quadrado central).
- O frontend usa React, TypeScript, Vite e TensorFlow.js e carrega `public/model/`. Publicado na Vercel (projeto `rota-segura`).
- O app é um protótipo acadêmico e deve manter avisos explícitos de segurança e limitações.
- Validar o frontend com `npm run build`; o Python usa 3.11/3.12.
