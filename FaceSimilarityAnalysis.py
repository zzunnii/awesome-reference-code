import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import insightface
from insightface.app import FaceAnalysis

# 🔹 1️⃣ 최신 얼굴 인식 모델 로드 (InsightFace)
app = FaceAnalysis(name="buffalo_l")  # 'buffalo_l' 모델은 고성능 ArcFace 기반
app.prepare(ctx_id=0, det_size=(640, 640))  # GPU 사용 가능하면 ctx_id=0 (없으면 -1)

# 🔹 2️⃣ 이미지 리스트 (사용자 제공)
image_paths = [
    '''
    동일 인물인지 확인할 이미지 그룹
    '''
]

# 🔹 3️⃣ 얼굴 특징 벡터(임베딩) 추출 함수
def extract_embedding(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    faces = app.get(image)  # 얼굴 감지 및 임베딩 추출

    if len(faces) == 0:
        return None  # 얼굴을 감지하지 못하면 None 반환

    return faces[0].embedding  # 첫 번째 얼굴의 특징 벡터 반환

# 🔹 4️⃣ 모든 이미지에서 얼굴 특징 벡터 추출
face_encodings = []
valid_image_paths = []

for img_path in image_paths:
    embedding = extract_embedding(img_path)
    if embedding is not None:
        face_encodings.append(embedding)
        valid_image_paths.append(img_path)

# 🔹 5️⃣ 얼굴 유사도 계산 (코사인 유사도)
if len(face_encodings) > 1:
    similarity_matrix = cosine_similarity(face_encodings)

    # 👉 유사도 행렬을 pandas DataFrame으로 변환
    df_similarity = pd.DataFrame(
        similarity_matrix,
        index=[f"Image {i+1}" for i in range(len(valid_image_paths))],
        columns=[f"Image {i+1}" for i in range(len(valid_image_paths))]
    )

    # 🔹 6️⃣ 유사도 시각화 (히트맵)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_similarity, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=df_similarity.columns, yticklabels=df_similarity.index)
    plt.title("Face Similarity Heatmap")
    plt.show()

    # 🔹 7️⃣ DBSCAN을 사용한 얼굴 그룹화 (동일 인물 클러스터링)
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(face_encodings)
    labels = clustering.labels_

    # 🔹 8️⃣ 동일 인물 그룹화 결과 저장 및 폴더 생성
    output_base_folder = #r"이미지 저장할 곳"
    os.makedirs(output_base_folder, exist_ok=True)

    group_dict = {}
    for idx, label in enumerate(labels):
        if label not in group_dict:
            group_dict[label] = []
        group_dict[label].append(valid_image_paths[idx])

    # 🔹 9️⃣ 그룹별 폴더 생성 후 파일 이동
    print("\n===== 얼굴 그룹화 및 파일 저장 =====")
    for group, images in group_dict.items():
        group_folder = os.path.join(output_base_folder, f"Group_{group}")
        os.makedirs(group_folder, exist_ok=True)  # 폴더 생성

        print(f"\n📂 [Group {group}] → 저장 위치: {group_folder}")
        for img in images:
            filename = os.path.basename(img)
            dest_path = os.path.join(group_folder, filename)
            shutil.copy(img, dest_path)  # 파일 복사
            print(f"✅ {filename} → {dest_path}")

    print("\n🎉 그룹화된 이미지들이 폴더에 저장되었습니다!")

else:
    print("얼굴을 감지할 수 없는 이미지가 많아서 유사도 분석이 어렵습니다.")
