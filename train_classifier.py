import json
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# ============================================
# CONFIG
# ============================================
DATASETS = [
    {
        'shots': r'input/data/2022_BCN_FinalF_1_shots.csv',
        'pose':  r'input/data/2022_BCN_FinalF_1_pose.json'
    },
    {
        'shots': r'input/data/2022_BCN_FinalM_1_shots.csv',
        'pose':  r'input/data/2022_BCN_FinalM_1_pose.json'
    }
]


def load_pose_data(pose_path):
    print(f"Loading pose: {pose_path}")
    with open(pose_path) as f:
        data = json.load(f)

    img_id_to_frame = {}
    for img in data['images']:
        frame_id = int(
            img['file_name'].split('_')[1].split('.')[0]
        )
        img_id_to_frame[img['id']] = frame_id

    frame_to_keypoints = {}
    for ann in data['annotations']:
        if ann['num_keypoints'] == 0:
            continue
        frame_id = img_id_to_frame.get(ann['image_id'])
        if frame_id is None:
            continue
        if frame_id not in frame_to_keypoints:
            frame_to_keypoints[frame_id] = ann['keypoints']

    print(f"Loaded {len(frame_to_keypoints)} pose frames")
    return frame_to_keypoints


def extract_features(keypoints):
    """
    Extract features from dataset keypoints.
    Dataset format: [x, y, visibility, x, y, v, ...]
    
    Key insight: normalize everything by shoulder width.
    This makes features scale invariant — same values
    whether using pixel coordinates (dataset) or
    normalized 0-1 coordinates (YOLOv8-pose inference).
    This eliminates the domain gap between training
    and inference!
    """
    def get_xy(idx):
        x = keypoints[idx*3]
        y = keypoints[idx*3+1]
        return x, y

    try:
        l_sx, l_sy = get_xy(5)   # left shoulder
        r_sx, r_sy = get_xy(6)   # right shoulder
        l_ex, l_ey = get_xy(7)   # left elbow
        r_ex, r_ey = get_xy(8)   # right elbow
        l_wx, l_wy = get_xy(9)   # left wrist
        r_wx, r_wy = get_xy(10)  # right wrist
        l_hx, l_hy = get_xy(11)  # left hip
        r_hx, r_hy = get_xy(12)  # right hip
    except IndexError:
        return None

    # Body reference points
    body_cx = (l_sx + r_sx) / 2
    body_cy = (l_sy + r_sy) / 2
    sw = abs(r_sx - l_sx) + 1e-6  # shoulder width

    hip_cx = (l_hx + r_hx) / 2
    hip_cy = (l_hy + r_hy) / 2

    # All features normalized by shoulder width
    # Scale invariant regardless of coordinate system
    return [
        # Wrist positions relative to body
        (r_wx - body_cx) / sw,
        (r_wy - body_cy) / sw,
        (l_wx - body_cx) / sw,
        (l_wy - body_cy) / sw,

        # Elbow positions relative to body
        (r_ex - body_cx) / sw,
        (r_ey - body_cy) / sw,
        (l_ex - body_cx) / sw,
        (l_ey - body_cy) / sw,

        # Wrist height above shoulder
        (r_sy - r_wy) / sw,
        (l_sy - l_wy) / sw,

        # Arm extension
        np.sqrt((r_wx-r_sx)**2 + (r_wy-r_sy)**2) / sw,
        np.sqrt((l_wx-l_sx)**2 + (l_wy-l_sy)**2) / sw,

        # Shoulder rotation angle
        np.arctan2(r_sy - l_sy, r_sx - l_sx),

        # Torso height
        abs(body_cy - hip_cy) / sw,

        # Wrist relative to hip
        (r_wx - hip_cx) / sw,
        (r_wy - hip_cy) / sw,

        # Elbow bend angles
        np.arctan2(r_wy - r_ey, r_wx - r_ex),
        np.arctan2(l_wy - l_ey, l_wx - l_ex)
    ]


# ============================================
# BUILD TRAINING DATA
# ============================================
print("=" * 50)
print("Building training data from both datasets...")
print("=" * 50)

X = []
y = []

for dataset in DATASETS:
    print(f"\nProcessing: {dataset['shots']}")

    df = pd.read_csv(dataset['shots'], sep=';')
    df = df[df['category'].isin(
        ['Forehand', 'Backhand', 'Smash']
    )].copy()
    df['frame_id'] = df['file_name'].apply(
        lambda x: int(x.split('_')[1].split('.')[0])
    )

    print(f"Labeled frames: {len(df)}")
    print(df['category'].value_counts())

    frame_to_kps = load_pose_data(dataset['pose'])

    matched = 0
    skipped = 0

    for _, row in df.iterrows():
        fid = row['frame_id']
        if fid not in frame_to_kps:
            skipped += 1
            continue
        try:
            features = extract_features(frame_to_kps[fid])
            if features is None:
                skipped += 1
                continue
            X.append(features)
            y.append(row['category'])
            matched += 1
        except Exception:
            skipped += 1
            continue

    print(f"Matched: {matched} | Skipped: {skipped}")

X = np.array(X)
y = np.array(y)

print(f"\n{'=' * 50}")
print(f"Total training samples: {len(X)}")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count}")
print('=' * 50)

# ============================================
# TRAIN CLASSIFIER
# ============================================
print("\nTraining Shot Classifier...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================
# SAVE MODELS
# ============================================
with open('models/shot_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('models/shot_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models saved to models/")
print("\nNow run: python main.py")