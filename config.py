import warnings
from enum import Enum, unique
warnings.filterwarnings('ignore')
import os
import torch
import logging
import platform
import stat
from fsplit.filesplit import Filesplit
import paddle

paddle.disable_signal_handler()
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama')
STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'infer_model.pth')
VIDEO_INPAINT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'video')
MODEL_VERSION = 'V4'
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')

# Verifica e une arquivos de modelo caso estejam divididos
if 'big-lama.pt' not in (os.listdir(LAMA_MODEL_PATH)):
    fs = Filesplit()
    fs.merge(input_dir=LAMA_MODEL_PATH)

if 'inference.pdiparams' not in os.listdir(DET_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=DET_MODEL_PATH)

if 'ProPainter.pth' not in os.listdir(VIDEO_INPAINT_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=VIDEO_INPAINT_MODEL_PATH)

# Configura o caminho do ffmpeg
sys_str = platform.system()
if sys_str == "Windows":
    ffmpeg_bin = os.path.join('win_x64', 'ffmpeg.exe')
elif sys_str == "Linux":
    ffmpeg_bin = os.path.join('linux_x64', 'ffmpeg')
else:
    ffmpeg_bin = os.path.join('macos', 'ffmpeg')
FFMPEG_PATH = os.path.join(BASE_DIR, '', 'ffmpeg', ffmpeg_bin)

if 'ffmpeg.exe' not in os.listdir(os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64')):
    fs = Filesplit()
    fs.merge(input_dir=os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64'))

os.chmod(FFMPEG_PATH, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

@unique
class InpaintMode(Enum):
    STTN = 'sttn'
    LAMA = 'lama'
    PROPAINTER = 'propainter'

# Configurações gerais
USE_H264 = True

# Configurações de modo de inpaint
MODE = InpaintMode.STTN  # Algoritmo escolhido
THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = 10  # Diferença de altura e largura para detectar legendas
SUBTITLE_AREA_DEVIATION_PIXEL = 20  # Expande a área do mask para evitar bordas remanescentes
THRESHOLD_HEIGHT_DIFFERENCE = 20  # Tolerância para definir se as caixas de texto pertencem à mesma linha
PIXEL_TOLERANCE_Y = 20  # Tolerância de pixels no eixo Y
PIXEL_TOLERANCE_X = 20  # Tolerância de pixels no eixo X

# Configurações específicas do modo STTN para melhor qualidade
STTN_SKIP_DETECTION = False  # Não pula a detecção de legendas para evitar remoção em frames sem legenda
STTN_NEIGHBOR_STRIDE = 5  # Mantém o stride baixo para maior precisão
STTN_REFERENCE_LENGTH = 15  # Aumenta o número de frames de referência para melhor qualidade
STTN_MAX_LOAD_NUM = 50  # Ajuste maior para melhorar os resultados, usando mais memória

# Verifica se STTN_MAX_LOAD_NUM é maior que a multiplicação de STTN_REFERENCE_LENGTH e STTN_NEIGHBOR_STRIDE
if STTN_MAX_LOAD_NUM < STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE:
    STTN_MAX_LOAD_NUM = STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE

# Configurações para modo PROPAINTER
PROPAINTER_MAX_LOAD_NUM = 70  # Ajustado para maior qualidade, mas requer memória maior

# Configurações para modo LAMA
LAMA_SUPER_FAST = False  # Desativa o modo rápido para garantir melhor qualidade
