# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from streamlit_paste_button import paste_image_button as pbutton

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

# 캐싱으로 Reader를 한 번만 로드
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False, verbose=False)

def extract_number(region, reader):
    try:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        results = reader.readtext(resized, allowlist='0123456789', paragraph=False)

        if results:
            text = results[0][1]
            return int(text.strip())
    except Exception as e:
        st.error(f"숫자 추출 오류: {e}")
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
            else:
                results[type] = 0
        else:
            results[type] = 0

    return results


def process_image(img):
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='업로드된 이미지', use_container_width=True)
    
    with st.spinner('아이템 개수를 세는 중... (처음 실행시 모델 로딩으로 시간이 걸릴 수 있습니다)'):
        try:
            reader = load_reader()
            results = find_items(img, color_ranges, reader)
            
            # 결과 표시
            st.success('✅ 분석 완료!')
            
            col1, col2, col3 = st.columns(3)
            
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
            
            cols = [col1, col2, col3]
            counts_str = ''
            for idx, (type, count) in enumerate(results.items()):
                col = cols[idx % 3]
                with col:
                    st.metric(
                        label=f"{emoji_map.get(type, '')} {korean_map.get(type, type)}", 
                        value=f"{count}개"
                    )
                counts_str += f"{count}/"
            st.markdown(f'{counts_str}')
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
            st.info("EasyOCR 모델 로딩에 실패했을 수 있습니다. Streamlit Cloud의 메모리 제한 때문일 수 있습니다.")


# Streamlit UI
st.set_page_config(page_title="게임 아이템 카운터", page_icon="🎮")

st.title('🎮 게임 아이템 카운터')
st.write('획득한 속성 아이템 개수를 자동으로 세어드립니다!')

tab1, tab2 = st.tabs(["📁 파일 업로드", "📋 붙여넣기"])

with tab1:
    st.write('이미지 파일을 업로드하세요')
    uploaded_file = st.file_uploader("스크린샷을 업로드하세요", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        process_image(img)

with tab2:
    st.write('이미지를 붙여넣으세요')
    paste_result = pbutton(
        label='이미지를 붙여넣으세요',
        background_color="#FF4B4B",
        hover_background_color="#FF6B6B",
    )

    if paste_result is not None:
        pil_image = paste_result.image_data
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        process_image(img)


st.markdown('---')
st.caption('Made by ❤️sseong')
