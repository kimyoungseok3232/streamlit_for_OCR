import json
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="데이터 분석 및 개별 이미지 확인용", layout="wide")

global language
language = ['chinese', 'japanese', 'thai', 'vietnamese']


# json 파일에서 각 key 별로 데이터 불러와서 dataframe으로 변환 후 리스트에 넣고 리스트 반환
# 입력 - json 파일
# 출력 - 데이터프레임 딕셔너리
def json_to_csv(filename):
    data = {}
    
    for key in filename:
        data[key] = {}
        df = {}
        filename_list = list(filename[key]['images'])
        df["file_name"] = filename_list
        annos = pd.DataFrame(columns=['transcription', 'points', 'image_id'])
        anno_num = []
        for name in filename_list:
            anno = pd.DataFrame(filename[key]['images'][name]['words']).transpose().reset_index(drop=True)
            anno_num.append(anno.shape[0])
            anno['image_id'] = name
            annos = pd.concat([annos,anno]).reset_index(drop=True)
        df["annotation_num"] = anno_num
        data[key]['images'] = pd.DataFrame(df)
        data[key]['annotations'] = annos
    return data

@st.cache_data
def load_json_data():
    train_data = {}
    test_data = {}
    for lang in language:
        with open(f'../data/{lang}_receipt/ufo/train.json', encoding="UTF-8") as t:
            train_data[lang] = json.loads(t.read())
        with open(f'../data/{lang}_receipt/ufo/test.json', encoding="UTF-8") as t:
            test_data[lang] = json.loads(t.read())
    test = json_to_csv(test_data)
    train = json_to_csv(train_data)
    
    return test, train
# 출력 - train, test 데이터프레임 딕셔너리

# 데이터 페이지 단위로 데이터프레임 스플릿
# 입력 - input_df(이미지 데이터), anno_df(박스 그리기 용), rows(한번에 보여줄 데이터 수)
# 출력 - df(이미지 데이터프레임 리스트), df2(박스 그리기 용 데이터프레임 리스트)
@st.cache_data()
def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df

# 페이지에 있는 이미지 출력
# 입력
## type = 이미지 경로 찾을 때 사용(../dataset/train/, ../dataset/test/ 이미지 서로 다른 폴더인 경우 사용 가능),
## img_pathes = train_data['image_id'] or test_data['image_id'] 데이터프레임
## anno = train_data[['bbox','category_id']] 데이터프레임
## window = 데이터 출력할 창
def get_image(lang, image_path, anno, transform, type):
    img = cv2.imread(f'../data/{lang}_receipt/img/{type}/'+image_path)
    tlist = [0 for _ in range(10)]
    tset = set()
    iters = anno[['points']].values

    for [annotation] in iters:
        cv2.polylines(img, [np.array(annotation, dtype=np.int32)],True, (255,0,0), 3)


    return img, tlist, tset

def show_images(lang, img_pathes, anno, window, type):
    cols = window.columns(3)
    for idx,[path,annot] in enumerate(img_pathes.values):
        if idx%3 == 0:
            cols = window.columns(3)

        img, tlist, tset = get_image(lang, path, anno[anno['image_id']==path], 0, type)

        cols[idx%3].image(img)
        cols[idx%3].write(img.shape)
        cols[idx%3].write(path)

# 데이터 프레임 페이지 단위로 출력
# 입력
## img = train_data or test_data에서 'images'
## anno = train_data or test_data에서 'annotations'
## window = 데이터 프레임 출력할 위치
## type = 이미지 경로
def show_dataframe(img,anno,window,lang,type):
    # 가장 윗부분 데이터 정렬할 지 선택, 정렬 시 무엇으로 정렬할지, 오름차순, 내림차순 선택
    top_menu = window.columns(3)
    with top_menu[0]:
        sort = st.radio("Sort Data", options=["Yes", "No"], horizontal=1, index=1, key=[lang,window,1])
    if sort == "Yes":
        with top_menu[1]:
            sort_field = st.selectbox("Sort By", options=img.columns, key=[lang,window,2])
        with top_menu[2]:
            sort_direction = st.radio(
                "Direction", options=["⬆️", "⬇️"], horizontal=True
            )
        img = img.sort_values(
            by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True
        )
    # 데이터 크기 출력
    total_data = img.shape
    with top_menu[0]:
        st.write("data_shape: "+str(total_data))
    con1,con2 = window.columns((1,3))

    # 아래 부분 페이지당 데이터 수, 페이지 선택
    bottom_menu = window.columns((4, 1, 1))
    with bottom_menu[2]:
        batch_size = st.selectbox("Page Size", options=[9, 15, 27], key=[lang,window,3])
    with bottom_menu[1]:
        total_pages = (
            int(len(img) / batch_size) if int(len(img) / batch_size) > 0 else 1
        )
        current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, step=1
        )
    with bottom_menu[0]:
        st.markdown(f"Page **{current_page}** of **{total_pages}** ")
    pages = split_frame(img, batch_size)
    if 'annotation_num' in pages[0].columns:
        con1.dataframe(data=pages[current_page - 1][['file_name','annotation_num']], use_container_width=True)
    else:
        con1.dataframe(data=pages[current_page - 1]['file_name'], use_container_width=True)

    show_images(lang, pages[current_page - 1], anno, con2, type)


def main():
    if st.sidebar.button("새로고침"):
        st.rerun()
    # 원본데이터 확인 가능 아웃풋도 확인하도록 할 수 있을 듯?
    option = st.sidebar.selectbox("데이터 선택",("이미지 데이터"))
    
    # 데이터 로드
    testd, traind = load_json_data()

    if option == "이미지 데이터":
        # 트레인 데이터 출력
        choose_data = st.sidebar.selectbox("트레인/테스트", ("train", "test"))
        choose_lang = st.sidebar.selectbox("언어 선택", language)

        if choose_data == "train":
            st.header("트레인 데이터")
            choose_type = st.sidebar.selectbox("시각화 선택", ("이미지 출력"))
            if choose_type == "이미지 출력":
                show_dataframe(traind[choose_lang]['images'],traind[choose_lang]['annotations'],st,choose_lang, choose_data)

        elif choose_data == "test":
            st.header("테스트 데이터")

            show_dataframe(testd[choose_lang]['images'],testd[choose_lang]['annotations'],st,choose_lang, choose_data)



def login(password, auth):
    if password in auth:
        st.session_state['login'] = True
    else:
        st.write('need password')

if 'login' not in st.session_state or st.session_state['login'] == False:
    auth = set(['T7157','T7122','T7148','T7134','T7104','T7119'])
    password = st.sidebar.text_input('password',type='password')
    button = st.sidebar.button('login',on_click=login(password, auth))

elif st.session_state['login'] == True:
    main()