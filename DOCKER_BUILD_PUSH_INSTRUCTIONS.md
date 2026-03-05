# Docker Build & Push Instructions

## Что уже подготовлено
1. `requirements.txt` конвертирован в UTF-8 (без BOM).
2. Добавлен `.dockerignore`.
3. Добавлены `Dockerfile.cpu` и `Dockerfile.gpu`.

## Пошаговые действия

1. Перейти в проект:

```bash
cd /home/al0n1/my/unik/pp/video_translator/PP_ML_PLATFORM
```

2. (Опционально) закоммитить изменения:

```bash
git add requirements.txt .dockerignore Dockerfile.cpu Dockerfile.gpu
git commit -m "Add CPU/GPU Dockerfiles and normalize requirements encoding"
```

3. Использовать отдельный `DOCKER_CONFIG` (обход ошибки `docker-credential-desktop`):

```bash
mkdir -p /tmp/docker-config-codex
printf '{"auths":{}}\n' > /tmp/docker-config-codex/config.json
```

4. Логин в Docker Hub:

```bash
DOCKER_CONFIG=/tmp/docker-config-codex docker login -u al0n1s
```

5. Подготовить `buildx` builder:

```bash
DOCKER_CONFIG=/tmp/docker-config-codex docker buildx create --name ppml-builder --use 2>/dev/null || \
DOCKER_CONFIG=/tmp/docker-config-codex docker buildx use ppml-builder

DOCKER_CONFIG=/tmp/docker-config-codex docker buildx inspect --bootstrap
```

6. Сборка и пуш CPU/GPU:

```bash
REPO=al0n1s/video-translator-ml-service
SHA=$(git rev-parse --short HEAD)

DOCKER_CONFIG=/tmp/docker-config-codex docker buildx build --platform linux/amd64 -f Dockerfile.cpu \
  -t $REPO:latest-cpu -t $REPO:${SHA}-cpu --push .

DOCKER_CONFIG=/tmp/docker-config-codex docker buildx build --platform linux/amd64 -f Dockerfile.gpu \
  -t $REPO:latest-gpu -t $REPO:${SHA}-gpu --push .
```

7. Проверка опубликованных тегов:

```bash
DOCKER_CONFIG=/tmp/docker-config-codex docker buildx imagetools inspect $REPO:latest-cpu
DOCKER_CONFIG=/tmp/docker-config-codex docker buildx imagetools inspect $REPO:latest-gpu
```

8. Smoke-тесты:

```bash
docker run --rm $REPO:latest-cpu python -c "import src.app; print('cpu-ok')"
docker run --rm --gpus all $REPO:latest-gpu python -c "import torch; print(torch.cuda.is_available())"
```

## Теги образов
- `al0n1s/video-translator-ml-service:latest-cpu`
- `al0n1s/video-translator-ml-service:latest-gpu`
- `al0n1s/video-translator-ml-service:<sha>-cpu`
- `al0n1s/video-translator-ml-service:<sha>-gpu`

