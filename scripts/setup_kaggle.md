
# Configuracion de Kaggle

1. Instala la CLI:

```bash
pip install kaggle
```

2. En Kaggle, entra a `Account > API > Create New Token`.

3. Guarda el archivo `kaggle.json` en:

```text
%USERPROFILE%\.kaggle\kaggle.json
```

4. Descarga el dataset del modulo 2:

```bash
python scripts/download_data.py --output-dir data/raw
```

El entrenamiento espera una estructura compatible con `torchvision.datasets.ImageFolder`, por ejemplo:

```text
data/raw/module2_distraction/
  train/
    safe_driving/
    texting/
  test/
    safe_driving/
    texting/
```

Si el dataset no trae carpetas `train`/`test`, el loader divide automaticamente las carpetas de clases en train, validacion y prueba.
