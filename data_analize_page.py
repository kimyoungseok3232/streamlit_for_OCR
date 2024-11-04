import json
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="데이터 분석 및 개별 이미지 확인용", layout="wide")

language = ['chinese', 'japanese', 'thai', 'vietnamese']
checker = ['.zh.', '.ja.', '.th.', '.vi.']

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
def get_image(lang, image_path, anno, transform, type, choose_anno = None):
    img = cv2.imread(f'../data/{lang}_receipt/img/{type}/'+image_path)
    if st.session_state['show_anno']:
        iters = anno[['points']].values
        for idx, [annotation] in enumerate(iters):
            if choose_anno == idx:
                cv2.polylines(img, [np.array(annotation, dtype=np.int32)],True, (0,255,0), 3)
            else:
                cv2.polylines(img, [np.array(annotation, dtype=np.int32)],True, (255,0,0), 3)
    elif choose_anno != None:
        iters = anno[['points']].values
        for idx, [annotation] in enumerate(iters):
            if choose_anno == idx:
                cv2.polylines(img, [np.array(annotation, dtype=np.int32)],True, (0,255,0), 3)
        
    return img

def show_images(lang, img_pathes, anno, window, type):
    cols = window.columns(3)
    for idx,[path,annot] in enumerate(img_pathes.values):
        if idx%3 == 0:
            cols = window.columns(3)
        img = get_image(lang, path, anno[anno['image_id']==path], 0, type)
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
            len(img) // batch_size+1 if len(img) % batch_size > 0 else len(img) // batch_size
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

def csv_list(output_dir):
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    return csv_files

def reorganize_data(data):
    # 결과를 저장할 새로운 데이터 구조
    new_data = {lang: {'images': {}} for lang in language}

    # 기존 데이터에서 file_name 기반으로 분류
    for file_name, info in data['images'].items():
        # checker에 있는 특정 문자열에 따라 언어 결정
        for idx, lang in enumerate(checker):
            if lang in file_name:
                language_key = language[idx]
                new_data[language_key]['images'][file_name] = info
                break
        else:
            # 위에 지정된 언어에 해당하지 않으면 'other'로 분류
            if 'other' not in new_data:
                new_data['other'] = {'images': {}}
            new_data['other']['images'][file_name] = info

    return new_data

@st.cache_data()
def csv_to_json(file_name):
    with open(f'./output/{file_name}') as f:
        data = json.loads(f.read())
    return json_to_csv(reorganize_data(data))

def check_same_csv(name, csv):
    i = 1
    while name in csv:
        if i == 1:
            name = name[:-4]+'_'+str(i)+'.csv'
        else:
            name = name[:-6]+'_'+str(i)+'.csv'
        i += 1
    return name

@st.dialog("csv upload")
def upload_csv(csv):
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    # 파일이 업로드되면 처리
    if uploaded_file is not None:
        data = json.load(uploaded_file)

        try:
            json_to_csv(reorganize_data(data))
            input_name = st.text_input("csv 파일 이름 지정", value=uploaded_file.name.replace('.csv', ''))

            if st.button("upload_csv"):
                name = check_same_csv(input_name+'.csv',csv)
                st.write("saved file name: "+name)
                with open(f'./output/{name}','w') as f:
                    json.dump(data, f, indent=4)
        except:
            st.error("비정상적인 파일 입니다.")
        if st.button("close"):
                st.rerun()

@st.cache_data()
def get_cropped_img(choose_lang, path, annodata, anno_num, point, show_point):
    ori_img = cv2.imread(f'../data/{choose_lang}_receipt/img/train/'+path)
    mask = np.zeros_like(ori_img)
    cv2.fillPoly(mask, [np.array(annodata[annodata['image_id']==path].iloc[anno_num-1]['points'],dtype=np.int32)], (255,255,255))
    masked_img = cv2.bitwise_and(ori_img, mask)
    x,y,w,h = cv2.boundingRect(np.array(annodata[annodata['image_id']==path].iloc[anno_num-1]['points'],dtype=np.int32))

    if show_point:
        cv2.circle(masked_img, np.array(point,dtype=np.int32), 5, (0,0,255), 3)

    cropped_img = masked_img[y:y+h, x:x+w]
    return cropped_img

def csv_to_backup(csv):
    if not os.path.exists('./backup/'):
        os.makedirs('./backup/')
    os.rename('./output/'+csv,'./backup/'+csv)
    st.rerun()


def main():
    if st.sidebar.button("새로고침"):
        st.rerun()
    # 원본데이터 확인 가능 아웃풋도 확인하도록 할 수 있을 듯?
    option = st.sidebar.selectbox("데이터 선택",("이미지 데이터","라벨링","backup"))
    
    # 데이터 로드
    testd, traind = load_json_data()

    if option == "이미지 데이터":
        # 트레인 데이터 출력
        choose_data = st.sidebar.selectbox("트레인/테스트", ("train", "test"))
        choose_lang = st.sidebar.selectbox("언어 선택", language)
        st.session_state['show_anno'] = st.sidebar.checkbox("어노테이션 표시", value=True)

        if choose_data == "train":
            st.header("트레인 데이터")
            choose_type = st.sidebar.selectbox("시각화 선택", ("이미지 출력"))
            if choose_type == "이미지 출력":
                show_dataframe(traind[choose_lang]['images'],traind[choose_lang]['annotations'],st,choose_lang, choose_data)

        elif choose_data == "test":
            st.header("테스트 데이터")
            if not os.path.exists('./output/'):
                os.makedirs('./output/')
            csv = csv_list('./output')
            choose_csv = st.sidebar.selectbox("output.csv적용",("안함",)+tuple(csv))
            data = testd
            if choose_csv != "안함":
                data = csv_to_json(choose_csv)
                if st.sidebar.button("현재 csv 백업 폴더로 이동"):
                    csv_to_backup(choose_csv)
            if st.sidebar.button("새로운 csv 파일 업로드"):
                upload_csv(csv)

            show_dataframe(data[choose_lang]['images'],data[choose_lang]['annotations'],st,choose_lang, choose_data)
    elif option == "라벨링":
        choose_lang = st.sidebar.selectbox("언어 선택", language)
        imgdata = traind[choose_lang]['images']
        if "change_anno" in st.session_state:
            annodata = st.session_state["change_anno"]
        else:
            annodata = traind[choose_lang]['annotations']
        image_num = st.sidebar.number_input("이미지 번호", min_value=1, max_value=imgdata.shape[0])
        path = imgdata.iloc[image_num-1]['file_name']
        anno_num = st.sidebar.number_input("bbox 번호", min_value=1, max_value=annodata[annodata['image_id']==path].shape[0])

        point_num = st.sidebar.number_input("꼭짓점 번호", min_value=0, max_value=3)
    
        st.session_state['show_anno'] = st.sidebar.checkbox("다른 어노테이션 표시", value=True)
        show_point = st.sidebar.checkbox("꼭짓점 표시", value=True)

        point = annodata[annodata['image_id']==path].iloc[anno_num-1]['points'][point_num]

        img = get_image(choose_lang, path, annodata[annodata['image_id']==path], 0, 'train', anno_num-1)
        if show_point:
            cv2.circle(img, np.array(point,dtype=np.int32), 5, (0,0,255), 3)

        cropped_img = get_cropped_img(choose_lang, path, annodata, anno_num, point, show_point)

        col1,col2 = st.columns(2)
        col1.image(img)
        col2.image(cropped_img)
        col2.write('bbox의 포인트 목록')
        col2.write(annodata[annodata['image_id']==path].iloc[anno_num-1]['points'])
        
        x = col2.number_input("x좌표", value=int(point[0]), step=1)
        y = col2.number_input("y좌표", value=int(point[1]), step=1)

        if col2.button("좌표 값 변경"):
            annodata[annodata['image_id']==path].iloc[anno_num-1]['points'][point_num] = [x,y]
            st.session_state["change_anno"] = annodata
            st.rerun()
        if col2.button("초기화"):
            annodata[annodata['image_id']==path].iloc[anno_num-1]['points'][point_num] = traind[choose_lang]['annotations'][annodata['image_id']==path].iloc[anno_num-1]['points'][point_num]
            st.session_state["change_anno"] = annodata
            st.rerun()
    elif option == "backup":
        st.header("backup 파일 목록")
        file_list = os.listdir('./backup/')
        for file in file_list:
            file_path = os.path.join('./backup/', file)
            if os.path.isfile(file_path):
                file_name, button1, button2 = st.columns([5,1,2])
                file_name.write(file)
                if button1.button("삭제", key = f"delete {file}"):
                    try:
                        os.remove(file_path)
                        st.success(f"{file} 파일이 삭제되었습니다.")
                    except:
                        st.error("파일 삭제 중 오류가 발생했습니다.")
                    st.rerun()
                if button2.button("기존 폴더로 이동", key = f"move {file}"):
                    try:
                        os.rename(file_path,'./output/'+file)
                        st.success(f"{file} 파일이 이동되었습니다.")
                    except:
                        st.error("파일 이동 중 오류가 발생했습니다.")
                    st.rerun()


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