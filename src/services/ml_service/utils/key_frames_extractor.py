import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Union, Dict


from src.services.ml_service.utils import Response, extract_frames


def extract_key_frames_histogram(frames: List[np.ndarray],
                                 threshold: float = 0.1) -> List[int]:
    key_frames = [0]
    for i in range(1, len(frames)):
        hist_prev = cv2.calcHist([frames[i-1]], [0, 1, 2], None, [8, 8, 8],
                                 [0, 256, 0, 256, 0, 256])
        hist_curr = cv2.calcHist([frames[i]], [0, 1, 2], None, [8, 8, 8],
                                 [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist_prev, hist_prev)
        cv2.normalize(hist_curr, hist_curr)
        diff = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA)
        if diff > threshold:
            key_frames.append(i)
    return key_frames

def extract_key_frames_diff(frames: List[np.ndarray],
                           threshold: float = 0.1) -> List[int]:
    key_frames = [0]
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i-1], frames[i])
        score = np.mean(diff) / 255.0  # нормализуем разницу в диапазоне [0, 1]
        if score > threshold:
            key_frames.append(i)
    return key_frames

def extract_keyframes_kmeans(frames: List[np.ndarray],
                             threshold: float = 0.1) -> List[int]:
    n_clusters = max(1, int(len(frames) * threshold))

    # Признак каждого кадра — нормализованная цветовая гистограмма
    features = []
    for frame in frames:
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256]).flatten()
        cv2.normalize(hist, hist)
        features.append(hist)
    features = np.array(features, dtype=np.float32)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features)

    # Из каждого кластера берём кадр, ближайший к центроиду
    key_frames = []
    for cluster_id in range(n_clusters):
        indices = np.where(labels == cluster_id)[0]
        dists = np.linalg.norm(features[indices] - kmeans.cluster_centers_[cluster_id], axis=1)
        key_frames.append(int(indices[np.argmin(dists)]))

    return sorted(key_frames)


class KeyFrameExtractor:
    def __init__(self, threshold: float = 0.1, extract_type: str = "diff"):
        self.threshold = threshold
        self.extract_type = extract_type
    
    def save_frames(self,
                    frames: List[np.ndarray],
                    save_folder: str) -> Response:
        if os.path.exists(save_folder) is False:
            return Response(False, "KeyFrameExtractor.save_frames. Save folder does not exist", None)
        try:
            for idx, frame in enumerate(frames):
                cv2.imwrite(os.path.join(save_folder, f"frame_{idx:016d}.jpg"), frame)
            return Response(True, None, None)
        except Exception as e:
            return Response(False, f"KeyFrameExtractor.save_frames. {str(e)}", None)
    
    def process(self,
                video_path: str,
                save_folder: str = "var/data_ocr/frames") -> Dict[str, Union[List[int], int]]:
        resp: Response = extract_frames(video_path)
        if not resp.status:
            return Response(False, f"KeyFrameExtractor.process. {resp.error}", None)
        frames = resp.result
        if self.extract_type == "histogram":
            key_frames = extract_key_frames_histogram(frames, self.threshold)
        elif self.extract_type == "diff":
            key_frames = extract_key_frames_diff(frames, self.threshold)
        elif self.extract_type == "kmeans":
            key_frames = extract_keyframes_kmeans(frames, self.threshold)   
        else:
            return Response(False, f"KeyFrameExtractor.process. Unsupported extract_type: {self.extract_type}", None)

        return Response(True, None, {"key_frames": key_frames, "total_frames": len(frames), "frames": frames})


if __name__ == "__main__":
    extractor = KeyFrameExtractor(threshold=0.1, extract_type="diff")
    result = extractor.process("var/data_ocr/small_sample.mp4")
    key_frames = result["key_frames"]
    total_frames = result["total_frames"]
    print("Key frame indices:", key_frames)
    print(f"Total key frames: {len(key_frames)}, Percentage: {len(key_frames) / total_frames * 100:.2f}%")
