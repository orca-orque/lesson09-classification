import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/orca-orque/data/main/stroke.csv"

st.set_page_config(page_title=f"성능 높이기 - {APP_TITLE}", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------- 상수
FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
BINARY = ("hypertension", "heart_disease")  # 0 또는 1만 있는 속성

MODEL_LOGIT = "로지스틱 회귀(확률로 답하는 모델)"
MODEL_TREE = "의사결정트리(질문으로 답하는 모델)"

NO_VALUE = "계산할 수 없음"
BLUE = "#1f77b4"  # 실제 뇌졸중 아님
RED = "#d62728"  # 실제 뇌졸중
BEFORE_COLOR = "#2ca02c"  # 가중치 맞추기 전
AFTER_COLOR = "#ff7f0e"  # 가중치 맞춘 뒤


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def count_cells(y_true, y_pred) -> dict:
    """뇌졸중(1)을 양성으로 두고 TP·FP·FN·TN 사람 수를 센다."""
    pairs = list(zip(list(y_true), list(y_pred)))
    return {
        "TP": sum(1 for a, p in pairs if a == 1 and p == 1),
        "FN": sum(1 for a, p in pairs if a == 1 and p == 0),
        "FP": sum(1 for a, p in pairs if a == 0 and p == 1),
        "TN": sum(1 for a, p in pairs if a == 0 and p == 0),
    }


def four_scores(c: dict) -> dict:
    """네 칸에서 정확도·재현율·정밀도·F1을 구한다. 나눌 수 없으면 None."""
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) else None  # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) else None  # 분모: 뇌졸중이라 예측한 사람 수
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"정확도": accuracy, "재현율": recall, "정밀도": precision, "F1": f1}


def percent(value) -> str:
    return NO_VALUE if value is None else f"{value * 100:.2f}%"


def before_after_table(before: dict, after: dict) -> pd.DataFrame:
    """네 칸과 네 지표를 '맞추기 전 / 맞춘 뒤'로 나란히 놓은 표."""
    s0, s1 = four_scores(before), four_scores(after)
    cell_rows = [
        ("TP · 찾아낸 환자", "TP"),
        ("FN · 놓친 환자", "FN"),
        ("FP · 헛짚은 사람", "FP"),
        ("TN · 아님을 맞힌 사람", "TN"),
    ]
    rows = [(label, f"{before[key]:,}명", f"{after[key]:,}명") for label, key in cell_rows]
    rows += [(name, percent(s0[name]), percent(s1[name])) for name in ["정확도", "재현율", "정밀도", "F1"]]
    return pd.DataFrame(rows, columns=["항목", "가중치 맞추기 전", "가중치 맞춘 뒤"])


def change_line(before: dict, after: dict) -> str:
    return (
        f"찾아낸 환자는 {before['TP']:,}명에서 {after['TP']:,}명으로, "
        f"놓친 환자는 {before['FN']:,}명에서 {after['FN']:,}명으로, "
        f"헛짚은 사람은 {before['FP']:,}명에서 {after['FP']:,}명으로 바뀌었어요."
    )


def jitter(values: list, step: float) -> list:
    """0/1 속성의 점이 겹치지 않도록 그림에서만 살짝 흩어 놓는다(계산에는 쓰지 않음)."""
    return [v + ((i * step) % 1 - 0.5) * 0.16 for i, v in enumerate(values)]


def describe_value(col: str, value: float) -> str:
    if col in BINARY:
        return f"{NAMES[col]} {int(round(value))}({'있음' if value > 0.5 else '없음'})"
    return f"{NAMES[col]} {value:.4g}"


# ---------------------------------------------------------------- 화면 시작
st.title("🚀 성능 높이기")
st.caption(
    "뇌졸중을 겪은 사람은 아주 드물어요. 모델이 ‘아님’ 쪽에 치우치지 않도록 두 쪽의 가중치를 맞추면 "
    "결과가 어떻게 달라지는지 살펴봐요. 뇌졸중(stroke가 1)을 양성으로 둬요."
)

try:
    raw = load_data()
except Exception as e:  # 인터넷 연결이나 주소 문제
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# 1) 입력 속성 고르기
chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: NAMES[c],
)
features = [c for c in FEATURES if c in chosen]  # 항상 같은 순서로 맞춘다

if len(features) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 그래야 모델과 그림을 만들 수 있어요.")
    st.stop()

# 2) 번호 순으로 정렬 → 열 명씩 묶기 → 각 묶음의 앞 세 명이 테스트용
df = raw.sort_values("id", kind="stable").reset_index(drop=True)
is_test = (df.index % 10) < 3
test_df = df[is_test]
train_df = df[~is_test]

X_train = train_df[features].copy()
X_test = test_df[features].copy()
y_train = train_df["stroke"]
y_test = test_df["stroke"]

st.caption(
    "번호(id) 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했어요. "
    f"테스트용 {len(test_df):,}명, 학습용 {len(train_df):,}명이에요."
)

# 체질량지수는 골랐을 때만, 훈련용의 중앙값으로 빈 값을 채운다.
if "bmi" in features:
    bmi_median = X_train["bmi"].median()
    n_missing = int(X_train["bmi"].isna().sum() + X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    st.caption(f"체질량지수의 빈 값 {n_missing:,}개는 훈련용 중앙값 {bmi_median:.2f}로 채웠어요.")

# 3) 크기 맞추기(훈련용으로만 계산)
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)

# 4) 가중치를 맞추기 전과 맞춘 뒤의 모델
logit_before = LogisticRegression(max_iter=1000).fit(Xs_train, y_train)
logit_after = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs_train, y_train)
tree_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
tree_after = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42, class_weight="balanced"
).fit(X_train, y_train)

n_pos = int((y_train == 1).sum())
n_neg = int((y_train == 0).sum())
st.caption(
    "‘가중치 맞추기’는 훈련용에서 뇌졸중인 사람 한 명을 뇌졸중이 아닌 사람 "
    f"약 {n_neg / n_pos:.1f}명만큼 중요하게 보도록 하는 설정이에요(class_weight=\"balanced\"). "
    "로지스틱 회귀는 크기를 맞춘 값, 의사결정트리는 원래 값으로 학습해요."
)

cells = {
    "logit_before": count_cells(y_test, logit_before.predict(Xs_test)),
    "logit_after": count_cells(y_test, logit_after.predict(Xs_test)),
    "tree_before": count_cells(y_test, tree_before.predict(X_test)),
    "tree_after": count_cells(y_test, tree_after.predict(X_test)),
}

# ---------------------------------------------------------------- 전·후 비교표
st.header("1. 가중치를 맞추기 전과 후")
n_test_stroke = int((y_test == 1).sum())
st.write(f"테스트용 사람은 {len(test_df):,}명이고, 그중 뇌졸중인 사람은 {n_test_stroke:,}명이에요.")

left, right = st.columns(2)
with left:
    st.subheader(MODEL_LOGIT)
    st.dataframe(
        before_after_table(cells["logit_before"], cells["logit_after"]),
        hide_index=True,
        use_container_width=True,
    )
    st.write(change_line(cells["logit_before"], cells["logit_after"]))
with right:
    st.subheader(MODEL_TREE)
    st.dataframe(
        before_after_table(cells["tree_before"], cells["tree_after"]),
        hide_index=True,
        use_container_width=True,
    )
    st.write(change_line(cells["tree_before"], cells["tree_after"]))

st.caption(
    "TP: 뇌졸중인데 뇌졸중이라 예측 · FN: 뇌졸중인데 아니라고 예측 · FP: 아닌데 뇌졸중이라 예측 · "
    "TN: 아닌데 아니라고 예측. "
    f"뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1을 ‘{NO_VALUE}’으로 적었고, "
    f"정밀도와 재현율이 모두 0이어서 F1의 분모가 0일 때도 ‘{NO_VALUE}’으로 적었어요."
)


# ---------------------------------------------------------------- 그림 준비
def axis_range(col: str) -> tuple:
    """그림의 범위: 테스트 데이터의 범위에 여유를 조금 더한다."""
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    pad = 0.15 if col in BINARY else (hi - lo) * 0.05
    return lo - pad, hi + pad


test_median = X_test.median()
idx_pos = list(logit_before.classes_).index(1)

# ---------------------------------------------------------------- 2차원 산점도
st.header("2. 두 속성으로 본 0.5 경계선")
st.caption("0.5 경계선은 두 로지스틱 회귀 모델이 ‘뇌졸중일 확률 = 0.5’라고 보는 자리예요. 의사결정트리는 확률로 답하지 않아 그리지 않았어요.")

col_a, col_b = st.columns(2)
with col_a:
    x_col = st.selectbox("가로축", features, index=0, format_func=lambda c: NAMES[c], key="x2d")
with col_b:
    y_options = [c for c in features if c != x_col]
    y_col = st.selectbox("세로축", y_options, index=0, format_func=lambda c: NAMES[c], key="y2d")

x_lo, x_hi = axis_range(x_col)
y_lo, y_hi = axis_range(y_col)
fixed = [c for c in features if c not in (x_col, y_col)]

N = 100
xs = [x_lo + (x_hi - x_lo) * i / (N - 1) for i in range(N)]
ys = [y_lo + (y_hi - y_lo) * i / (N - 1) for i in range(N)]
grid = pd.DataFrame([(x, y) for y in ys for x in xs], columns=[x_col, y_col])
for c in fixed:
    grid[c] = float(test_median[c])
grid = grid[features]
grid_scaled = scaler.transform(grid)

fig2 = go.Figure()

# 테스트 데이터 점 (실제 뇌졸중 여부로 색 구분)
for flag, color, label in [(0, BLUE, "실제 뇌졸중 아님"), (1, RED, "실제 뇌졸중")]:
    mask = y_test == flag
    px_vals = X_test.loc[mask, x_col].tolist()
    py_vals = X_test.loc[mask, y_col].tolist()
    shown_x = jitter(px_vals, 0.6180339887) if x_col in BINARY else px_vals
    shown_y = jitter(py_vals, 0.7548776662) if y_col in BINARY else py_vals
    fig2.add_trace(
        go.Scatter(
            x=shown_x,
            y=shown_y,
            mode="markers",
            name=label,
            marker=dict(color=color, size=6, opacity=0.65),
            customdata=[[a, b] for a, b in zip(px_vals, py_vals)],
            hovertemplate=(
                f"{NAMES[x_col]}: %{{customdata[0]}}<br>"
                f"{NAMES[y_col]}: %{{customdata[1]}}<extra>%{{fullData.name}}</extra>"
            ),
        )
    )

# 두 모델의 0.5 경계선 겹쳐 그리기
outside_notes = []
for model, color, dash, label in [
    (logit_before, BEFORE_COLOR, "solid", "가중치 맞추기 전 0.5 선"),
    (logit_after, AFTER_COLOR, "dash", "가중치 맞춘 뒤 0.5 선"),
]:
    proba = model.predict_proba(grid_scaled)[:, idx_pos]
    p_min, p_max = float(proba.min()), float(proba.max())
    if p_min <= 0.5 <= p_max:
        rows = [proba[i * N:(i + 1) * N].tolist() for i in range(N)]
        fig2.add_trace(
            go.Contour(
                x=xs,
                y=ys,
                z=rows,
                contours=dict(start=0.5, end=0.5, size=1, coloring="lines", showlabels=False),
                colorscale=[[0, color], [1, color]],
                line=dict(width=3, dash=dash),
                showscale=False,
                showlegend=True,
                name=label,
                hoverinfo="skip",
            )
        )
    else:
        where = "낮아요" if p_max < 0.5 else "높아요"
        outside_notes.append(
            f"{label}은 이 그림 밖에 있어요. 그림 안에서는 모든 곳에서 뇌졸중일 확률이 0.5보다 {where} "
            f"(확률 범위 {p_min:.3f} ~ {p_max:.3f})."
        )

fig2.update_layout(
    title=f"테스트 데이터: {NAMES[x_col]} × {NAMES[y_col]}",
    xaxis_title=NAMES[x_col],
    yaxis_title=NAMES[y_col],
    xaxis=dict(range=[x_lo, x_hi]),
    yaxis=dict(range=[y_lo, y_hi]),
    height=580,
    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0),
)
st.plotly_chart(fig2, use_container_width=True)

if x_col in BINARY or y_col in BINARY:
    st.caption("고혈압·심장병처럼 0과 1만 있는 속성은 점이 겹치지 않게 그림에서만 살짝 흩어 찍었어요.")
if fixed:
    stand = ", ".join(describe_value(c, float(test_median[c])) for c in fixed)
    st.write(f"그림에 쓰지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {stand}")
else:
    st.write("고른 속성이 가로축·세로축 둘뿐이라 따로 세워 둔 값은 없어요.")
for note in outside_notes:
    st.write(note)

# ---------------------------------------------------------------- 3차원 산점도
st.header("3. 세 속성으로 본 0.5 평면")

if len(features) < 3:
    st.info("입력 속성을 세 개 이상 고르면 마우스로 돌려 볼 수 있는 3차원 그림이 나와요.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    a_col = st.selectbox("가로축", features, index=0, format_func=lambda c: NAMES[c], key="x3d")
with c2:
    b_options = [c for c in features if c != a_col]
    b_col = st.selectbox("세로축", b_options, index=0, format_func=lambda c: NAMES[c], key="y3d")
with c3:
    z_options = [c for c in features if c not in (a_col, b_col)]
    z_col = st.selectbox("높이축", z_options, index=0, format_func=lambda c: NAMES[c], key="z3d")

axes3 = [a_col, b_col, z_col]
ranges3 = [axis_range(c) for c in axes3]
fixed3 = [c for c in features if c not in axes3]


def plane_surface(model, color: str, label: str):
    """로지스틱 회귀의 0.5 평면을 원래 단위의 곡면으로 만든다. 그림 안에 없으면 None."""
    coef = model.coef_[0].tolist()
    mean = scaler.mean_.tolist()
    scale = scaler.scale_.tolist()
    w = [coef[j] / scale[j] for j in range(len(features))]  # 원래 단위에서의 가중치
    b = float(model.intercept_[0]) - sum(coef[j] * mean[j] / scale[j] for j in range(len(features)))

    axis_idx = [features.index(c) for c in axes3]
    const = b + sum(
        w[j] * float(test_median[features[j]]) for j in range(len(features)) if j not in axis_idx
    )
    wa = [w[i] for i in axis_idx]
    k = max(range(3), key=lambda i: abs(wa[i]))  # 가장 영향이 큰 축에 대해 풀어서 평면을 만든다
    if abs(wa[k]) < 1e-12:
        return None
    o0, o1 = [i for i in range(3) if i != k]

    M = 30
    u = [ranges3[o0][0] + (ranges3[o0][1] - ranges3[o0][0]) * i / (M - 1) for i in range(M)]
    v = [ranges3[o1][0] + (ranges3[o1][1] - ranges3[o1][0]) * i / (M - 1) for i in range(M)]
    mats = [[[None] * M for _ in range(M)] for _ in range(3)]
    any_inside = False
    for r, vv in enumerate(v):
        for q, uu in enumerate(u):
            kk = -(const + wa[o0] * uu + wa[o1] * vv) / wa[k]
            if ranges3[k][0] <= kk <= ranges3[k][1]:
                any_inside = True
                mats[o0][r][q] = uu
                mats[o1][r][q] = vv
                mats[k][r][q] = kk
    if not any_inside:
        return None
    return go.Surface(
        x=mats[0],
        y=mats[1],
        z=mats[2],
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=0.45,
        showlegend=True,
        name=label,
        hoverinfo="skip",
    )


fig3 = go.Figure()
steps = [0.6180339887, 0.7548776662, 0.5698402910]
for flag, color, label in [(0, BLUE, "실제 뇌졸중 아님"), (1, RED, "실제 뇌졸중")]:
    mask = y_test == flag
    real = [X_test.loc[mask, c].tolist() for c in axes3]
    shown = [jitter(real[i], steps[i]) if axes3[i] in BINARY else real[i] for i in range(3)]
    fig3.add_trace(
        go.Scatter3d(
            x=shown[0],
            y=shown[1],
            z=shown[2],
            mode="markers",
            name=label,
            marker=dict(color=color, size=3, opacity=0.7),
            customdata=[[a, b, c] for a, b, c in zip(real[0], real[1], real[2])],
            hovertemplate=(
                f"{NAMES[axes3[0]]}: %{{customdata[0]}}<br>"
                f"{NAMES[axes3[1]]}: %{{customdata[1]}}<br>"
                f"{NAMES[axes3[2]]}: %{{customdata[2]}}<extra>%{{fullData.name}}</extra>"
            ),
        )
    )

outside3 = []
for model, color, label in [
    (logit_before, BEFORE_COLOR, "가중치 맞추기 전 0.5 평면"),
    (logit_after, AFTER_COLOR, "가중치 맞춘 뒤 0.5 평면"),
]:
    surface = plane_surface(model, color, label)
    if surface is None:
        outside3.append(f"{label}은 이 그림 밖에 있어서 그리지 못했어요.")
    else:
        fig3.add_trace(surface)

fig3.update_layout(
    title=f"테스트 데이터: {NAMES[axes3[0]]} × {NAMES[axes3[1]]} × {NAMES[axes3[2]]}",
    scene=dict(
        xaxis=dict(title=NAMES[axes3[0]], range=list(ranges3[0])),
        yaxis=dict(title=NAMES[axes3[1]], range=list(ranges3[1])),
        zaxis=dict(title=NAMES[axes3[2]], range=list(ranges3[2])),
    ),
    height=680,
    legend=dict(orientation="h", yanchor="top", y=0.0, x=0),
)
st.plotly_chart(fig3, use_container_width=True)
st.caption("마우스로 끌면 돌려 볼 수 있고, 휠로 확대·축소할 수 있어요.")

if any(c in BINARY for c in axes3):
    st.caption("고혈압·심장병처럼 0과 1만 있는 속성은 점이 겹치지 않게 그림에서만 살짝 흩어 찍었어요.")
if fixed3:
    stand3 = ", ".join(describe_value(c, float(test_median[c])) for c in fixed3)
    st.write(f"그림에 쓰지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {stand3}")
else:
    st.write("고른 속성이 축 셋뿐이라 따로 세워 둔 값은 없어요.")
for note in outside3:
    st.write(note)
