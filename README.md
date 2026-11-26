# VLA Unitree G1 | Estação de logística

Protótipo de uma estação industrial genérica para uma apresentação de logística. O G1 recebe a instrução em português, identifica a caixa azul ou marrom pela câmera do MuJoCo e a transfere para a bancada de saída. As duas caixas mudam de posição entre episódios.

A demonstração usa uma **política modular experimental**. Um parser reconhece a cor solicitada, um detector de cor localiza o objeto em RGB e uma regressão treinada com exemplos visuais prevê a posição de aproximação da mão. A sequência de ações usa cinemática inversa. A base do G1 está fixa e a caixa acompanha a mão durante a pegada assistida. Após a soltura, a caixa é simulada com gravidade e contato. A cena não representa uma fábrica real nem testa força de pegada, equilíbrio ou locomoção.

O protótipo implementa a cadeia visão, linguagem e ação com componentes pequenos e interpretáveis. **SmolVLA ainda não está conectada ou treinada.** Os rótulos usados para calibrar a regressão são posições verdadeiras do simulador. A escolha da caixa e a predição do alvo durante a execução usam a instrução e a câmera RGB. O estado real da caixa é usado para inicialização, regra de contato da pegada assistida e avaliação.

## Rodar no Ubuntu-MuJoCo

Com o ambiente já usado para rodar o G1 e o SO-101:

```bash
cd ~
git clone https://github.com/MoronaCzech1991/VLA_UNITREE.git
cd VLA_UNITREE
source ~/mujoco-demo/.venv/bin/activate

env -u LIBGL_ALWAYS_SOFTWARE \
  GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  python demo.py --assets ~/mujoco-demo/mujoco_menagerie/unitree_g1 \
  --device cuda:0 --instruction "Pegue a caixa azul e coloque na saída"
```

Substitua **azul** por **marrom** para escolher a outra caixa. Passe `--seed 43` para mudar a distribuição inicial. Feche a janela e rode novamente para repetir a demonstração. O modo CUDA usa MuJoCo Warp para os passos de física, com cinemática inversa e decisão visual na CPU. O modo gráfico abre uma janela e deixa o resultado visível depois da entrega.

## Instalação independente

Requer Python 3.12 ou superior e Git. Em uma instalação nova:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git assets
git -C assets sparse-checkout set unitree_g1
```

Use `--assets assets/unitree_g1` nesse caso. O modelo original e suas malhas são distribuídos pelo [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1). Para rodar em CPU, passe `--device cpu`.

## Gerar exemplos e testar

Sem janela, use EGL se estiver disponível. O resultado inclui contagem de sucessos. O critério é a caixa selecionada apoiada na saída e a outra permanecendo na entrada.

```bash
MUJOCO_GL=egl python demo.py --assets ~/mujoco-demo/mujoco_menagerie/unitree_g1 \
  --device cpu --train --dataset runs/calibracao.npz --headless --episodes 8 \
  --instruction "Pegue a caixa marrom"
```

O NPZ opcional contém imagens RGB, instruções, características visuais e posições de aproximação. É um conjunto de calibração visual, sem trajetórias de ação no formato LeRobot. Para adaptar a SmolVLA, ainda é preciso registrar observações e ações a cada passo, converter os episódios para LeRobot e treinar ou ajustar um checkpoint para o espaço de ações do G1.
