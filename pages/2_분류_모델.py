import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/orca-orque/data/main/stroke.csv"

st.set_page_config(page_title=f"분류 모델 - {APP_TITLE}", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------- 상수
FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]  # bmi는 뺌
BINARY = ("hypertension", "heart_disease")  # 0 또는 1만 있는 속성

# 의사결정트리가 던지는 질문 ("예"는 항상 왼쪽 가지)
QUESTIONS = {
    "age": "나이가 {t} 이하인가?",
    "avg_glucose_level": "평균 혈당이 {t} 이하인가?",
    "bmi": "체질량지수가 {t} 이하인가?",
    "hypertension": "고혈압이 없는가?",
    "heart_disease": "심장병이 없는가?",
}

MODEL_LOGIT = "로지스틱 회귀(확률로 답하는 모델)"
MODEL_TREE = "의사결정트리(질문으로 답하는 모델)"
MODEL_BASE = "기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

BLUE = "#1f77b4"
RED = "#d62728"
GREEN = "#2ca02c"
BLUE_LIGHT = "rgba(66, 133, 244, 0.18)"
RED_LIGHT = "rgba(234, 67, 53, 0.22)"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def accuracy_card(title: str, test_acc: float, train_acc: float) -> None:
    """테스트 정확도를 크게, 훈련·테스트 정확도를 아래에 나란히 작게 보여 주는 카드."""
    html = (
        '<div style="text-align:center">'
        f'<div style="font-size:1rem; font-weight:600; min-height:3rem">{title}</div>'
        f'<div style="font-size:3rem; font-weight:700; line-height:1.2">{test_acc * 100:.2f}%</div>'
        '<div style="font-size:0.8rem; opacity:0.7">테스트 데이터 정확도</div>'
        '<div style="display:flex; justify-content:space-around; font-size:0.85rem; '
        'opacity:0.8; margin-top:0.6rem">'
        f"<span>훈련 {train_acc * 100:.2f}%</span>"
        f"<span>테스트 {test_acc * 100:.2f}%</span>"
        "</div></div>"
    )
    with st.container(border=True):
        st.markdown(html, unsafe_allow_html=True)


def jitter(values: list, step: float) -> list:
    """0/1 속성의 점이 겹치지 않도록 그림에서만 살짝 흩어 놓는다(계산에는 쓰지 않음)."""
    return [v + ((i * step) % 1 - 0.5) * 0.16 for i, v in enumerate(values)]


def describe_value(col: str, value: float) -> str:
    if col in BINARY:
        return f"{NAMES[col]} {int(round(value))}({'있음' if value > 0.5 else '없음'})"
    return f"{NAMES[col]} {value:.4g}"


def build_tree_dot(model: DecisionTreeClassifier, features: list) -> str:
    """학습된 의사결정트리를 graphviz DOT 문자열로 바꾼다."""
    t = model.tree_
    classes = list(model.classes_)
    idx_pos = classes.index(1)

    lines = [
        "digraph Tree {",
        "rankdir=LR;",
        'bgcolor="transparent";',
        'node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=12, '
        'fontcolor="#111111", color="#555555", margin="0.2,0.1", width=2.9];',
        'edge [color="#888888", fontcolor="#888888", fontname="sans-serif", fontsize=13];',
    ]

    for node in range(t.node_count):
        n = int(t.n_node_samples[node])
        values = t.value[node][0]
        # 버전에 따라 value가 사람 수일 수도, 비율일 수도 있어서 비율로 바꿔 다시 센다.
        pos = int(round(values[idx_pos] / values.sum() * n))
        ratio = pos / n * 100 if n else 0.0
        stats = f"훈련용 사람 {n:,}명\\n뇌졸중 {pos:,}명 ({ratio:.1f}%)"

        is_leaf = t.children_left[node] == -1
        if is_leaf:
            answer = classes[int(values.argmax())]
            answer_text = "뇌졸중" if answer == 1 else "아님"
            fill = "#f4c7c3" if answer == 1 else "#cfe2f3"
            label = f"답: {answer_text}\\n{stats}"
        else:
            feature = features[int(t.feature[node])]
            question = QUESTIONS[feature].format(t=f"{t.threshold[node]:.1f}")
            fill = "#f2f2f2"
            label = f"{question}\\n{stats}"
        lines.append(f'n{node} [label="{label}", fillcolor="{fill}"];')

    for node in range(t.node_count):
        if t.children_left[node] != -1:
            lines.append(f'n{node} -> n{t.children_left[node]} [label="예"];')
            lines.append(f'n{node} -> n{t.children_right[node]} [label="아니요"];')

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 화면 시작
st.title("🤖 분류 모델")
st.caption("뇌졸중(stroke가 1)을 양성으로 두고, 두 가지 모델이 뇌졸중 여부를 얼마나 맞히는지 살펴봐요.")

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

# 2) 테스트용 고정 나누기: 번호 순으로 정렬 → 열 명씩 묶기 → 각 묶음의 앞 세 명이 테스트용
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
    st.caption(
        f"체질량지수의 빈 값 {n_missing:,}개는 훈련용 중앙값 {bmi_median:.2f}로 채웠어요."
    )

# 3) 크기 맞추기(훈련용으로만 계산)
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)

# 4) 모델 만들기
logit = LogisticRegression(max_iter=1000).fit(Xs_train, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
baseline = DummyClassifier(strategy="most_frequent").fit(Xs_train, y_train)

st.caption(
    "로지스틱 회귀와 기준 모델은 크기를 맞춘 값으로 학습하고, "
    "의사결정트리는 질문의 기준 값을 원래 단위로 읽을 수 있게 크기를 맞추지 않은 값으로 학습했어요."
)

# ---------------------------------------------------------------- 정확도 카드
st.header("1. 정확도 비교")
c1, c2, c3 = st.columns(3)
with c1:
    accuracy_card(MODEL_LOGIT, logit.score(Xs_test, y_test), logit.score(Xs_train, y_train))
with c2:
    accuracy_card(MODEL_TREE, tree.score(X_test, y_test), tree.score(X_train, y_train))
with c3:
    accuracy_card(MODEL_BASE, baseline.score(Xs_test, y_test), baseline.score(Xs_train, y_train))

# ---------------------------------------------------------------- 산점도
st.header("2. 두 속성으로 본 모델의 경계")

col_a, col_b = st.columns(2)
with col_a:
    x_col = st.selectbox("가로축", features, index=0, format_func=lambda c: NAMES[c])
with col_b:
    y_options = [c for c in features if c != x_col]
    y_col = st.selectbox("세로축", y_options, index=0, format_func=lambda c: NAMES[c])

# 그림의 범위: 테스트 데이터의 범위에 여유를 조금 더한다.
def axis_range(col: str) -> tuple:
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    pad = 0.15 if col in BINARY else (hi - lo) * 0.05
    return lo - pad, hi + pad


x_lo, x_hi = axis_range(x_col)
y_lo, y_hi = axis_range(y_col)

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 둔다.
test_median = X_test.median()
fixed = [c for c in features if c not in (x_col, y_col)]

# 그림 전체를 덮는 격자를 만들어 두 모델에게 물어본다.
N = 100
xs = [x_lo + (x_hi - x_lo) * i / (N - 1) for i in range(N)]
ys = [y_lo + (y_hi - y_lo) * i / (N - 1) for i in range(N)]
grid = pd.DataFrame([(x, y) for y in ys for x in xs], columns=[x_col, y_col])
for c in fixed:
    grid[c] = float(test_median[c])
grid = grid[features]

idx_pos_logit = list(logit.classes_).index(1)
proba = logit.predict_proba(scaler.transform(grid))[:, idx_pos_logit]
proba_rows = [proba[i * N:(i + 1) * N].tolist() for i in range(N)]

tree_pred = (tree.predict(grid) == 1).astype(int)
tree_rows = [tree_pred[i * N:(i + 1) * N].tolist() for i in range(N)]

fig = go.Figure()

# (a) 의사결정트리가 나눈 칸 (옅은 색)
fig.add_trace(
    go.Heatmap(
        x=xs,
        y=ys,
        z=tree_rows,
        zmin=0,
        zmax=1,
        colorscale=[[0, BLUE_LIGHT], [1, RED_LIGHT]],
        showscale=False,
        hoverinfo="skip",
    )
)

# (b) 로지스틱 회귀가 0.5로 가르는 선
line_inside = float(proba.min()) <= 0.5 <= float(proba.max())
if line_inside:
    fig.add_trace(
        go.Contour(
            x=xs,
            y=ys,
            z=proba_rows,
            contours=dict(start=0.5, end=0.5, size=1, coloring="lines", showlabels=False),
            colorscale=[[0, GREEN], [1, GREEN]],
            line=dict(width=3),
            showscale=False,
            showlegend=True,
            name="로지스틱 회귀의 0.5 선",
            hoverinfo="skip",
        )
    )

# (c) 테스트 데이터 점 (실제 뇌졸중 여부로 색 구분)
for flag, color, label in [(0, BLUE, "실제 뇌졸중 아님"), (1, RED, "실제 뇌졸중")]:
    mask = y_test == flag
    px_vals = X_test.loc[mask, x_col].tolist()
    py_vals = X_test.loc[mask, y_col].tolist()
    shown_x = jitter(px_vals, 0.6180339887) if x_col in BINARY else px_vals
    shown_y = jitter(py_vals, 0.7548776662) if y_col in BINARY else py_vals
    fig.add_trace(
        go.Scatter(
            x=shown_x,
            y=shown_y,
            mode="markers",
            name=label,
            marker=dict(color=color, size=6, opacity=0.7),
            customdata=[[a, b] for a, b in zip(px_vals, py_vals)],
            hovertemplate=(
                f"{NAMES[x_col]}: %{{customdata[0]}}<br>"
                f"{NAMES[y_col]}: %{{customdata[1]}}<extra>%{{fullData.name}}</extra>"
            ),
        )
    )

# (d) 옅은 색 칸의 뜻을 알려 주는 범례
for value, color, label in [(0, BLUE_LIGHT, "나무가 '아님'이라 답하는 칸"), (1, RED_LIGHT, "나무가 '뇌졸중'이라 답하는 칸")]:
    if (tree_pred == value).any():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(symbol="square", size=14, color=color),
                name=label,
            )
        )

fig.update_layout(
    title=f"테스트 데이터: {NAMES[x_col]} × {NAMES[y_col]}",
    xaxis_title=NAMES[x_col],
    yaxis_title=NAMES[y_col],
    xaxis=dict(range=[x_lo, x_hi]),
    yaxis=dict(range=[y_lo, y_hi]),
    height=580,
    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0),
)
st.plotly_chart(fig, use_container_width=True)

if x_col in BINARY or y_col in BINARY:
    st.caption("고혈압·심장병처럼 0과 1만 있는 속성은 점이 겹치지 않게 그림에서만 살짝 흩어 찍었어요.")

if fixed:
    stand = ", ".join(describe_value(c, float(test_median[c])) for c in fixed)
    st.write(f"그림에 쓰지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {stand}")
else:
    st.write("고른 속성이 가로축·세로축 둘뿐이라 따로 세워 둔 값은 없어요.")

if not line_inside:
    if float(proba.max()) < 0.5:
        side = "모든 곳에서 뇌졸중일 확률이 0.5보다 낮아요"
    else:
        side = "모든 곳에서 뇌졸중일 확률이 0.5보다 높아요"
    st.write(
        f"로지스틱 회귀의 0.5 선은 이 그림 밖에 있어요. 그림 안에서는 {side} "
        f"(그림 안의 확률은 {float(proba.min()):.3f} ~ {float(proba.max()):.3f})."
    )

# ---------------------------------------------------------------- 가지 그림
st.header("3. 의사결정트리가 던진 질문")
st.caption("‘예’는 질문이 맞을 때, ‘아니요’는 아닐 때 가는 가지예요. 숫자는 훈련용 사람 수예요.")

st.graphviz_chart(build_tree_dot(tree, features), use_container_width=True)

t = tree.tree_
classes = list(tree.classes_)
leaves = [i for i in range(t.node_count) if t.children_left[i] == -1]
n_no = sum(1 for i in leaves if classes[int(t.value[i][0].argmax())] == 0)

used_idx = {int(f) for f in t.feature if f >= 0}
asked = [NAMES[c] for i, c in enumerate(features) if i in used_idx]
not_asked = [NAMES[c] for i, c in enumerate(features) if i not in used_idx]

st.write(f"답을 내는 마디는 모두 {len(leaves)}칸이고, 그중 {n_no}칸이 ‘아님’이라고 답해요.")
if asked:
    tail = f" (묻지 않은 것: {', '.join(not_asked)})" if not_asked else ""
    st.write(f"고른 속성 가운데 이 나무가 실제로 물은 것은 {', '.join(asked)}예요.{tail}")
else:
    st.write("이 나무는 질문을 하나도 던지지 않고 처음부터 답을 냈어요.")
