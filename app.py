# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import io

color_ranges = {
    'fire': 'B50B0E',
    'water': '015AB6',
    'wind': '1F6A0B',
    'earth': '623F23',
    'light': 'DA8D09',
    'dark': '502181'
}

def hex_to_rgb(hex_code):
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False, verbose=False)

def extract_number(region, reader):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(resized, allowlist='0123456789', paragraph=False)

    if results:
        text = results[0][1]
        try:
            return int(text.strip())
        except:
            return 0
    return 0

def find_items(img_array, color_range, reader):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    results = {}

    for type, hex_color in color_range.items():
        target_rgb = hex_to_rgb(hex_color)
        lower_c = np.array([max(0, c - 10) for c in target_rgb])
        upper_c = np.array([min(255, c + 10) for c in target_rgb])
        
        mask = cv2.inRange(img_rgb, lower_c, upper_c)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                x, y, w, h = cv2.boundingRect(largest)
                number_region = img_rgb[y+h:y+int(h*2), x+w:x+int(w*2.3)]
                count = extract_number(number_region, reader)
                results[type] = count

    return results

# Streamlit UI
st.title('🎮 게임 아이템 카운터')
st.write('획득한 속성 아이템 개수를 자동으로 세어드립니다!')

uploaded_file = st.file_uploader("스크린샷을 업로드하세요", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # 이미지 로드
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 이미지 표시
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='업로드된 이미지', use_column_width=True)
    
    with st.spinner('아이템 개수를 세는 중...'):
        reader = load_reader()
        results = find_items(img, color_ranges, reader)
    
    # 결과 표시
    st.success('분석 완료!')
    
    col1, col2 = st.columns(2)
    
    emoji_map = {
        'fire': '🔥',
        'water': '💧',
        'wind': '💨',
        'earth': '🌍',
        'light': '✨',
        'dark': '🌙'
    }
    
    korean_map = {
        'fire': '불',
        'water': '물',
        'wind': '바람',
        'earth': '대지',
        'light': '빛',
        'dark': '어둠'
    }
    
    for idx, (type, count) in enumerate(results.items()):
        col = col1 if idx % 2 == 0 else col2
        with col:
            st.metric(
                label=f"{emoji_map.get(type, '')} {korean_map.get(type, type)}", 
                value=f"{count}개"
            )

st.markdown('---')
st.caption('Made with ❤️ by SSEONG')