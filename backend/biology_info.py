# biology_info.py
biology_data = {
    "heron": {
        "name": "왜가리",
        "description": "한국에서 흔히 볼 수 있는 대형 조류로, 논이나 하천에서 가만히 서서 물고기를 기다리는 모습이 특징입니다.",
        "habitat": "하천, 호수, 논"
    },
    "mallard": {
        "name": "청둥오리",
        "description": "수컷은 머리가 화려한 녹색 빛을 띠며, 겨울철 하천에서 가장 흔하게 볼 수 있는 오리입니다.",
        "habitat": "강, 저수지, 해안"
    },
    "magpie": {
        "name": "까치",
        "description": "반가운 손님이 오면 운다는 길조로 알려져 있으며, 흑백의 깃털과 긴 꼬리가 특징입니다.",
        "habitat": "마을 부근, 고산지대를 제외한 전역"
    },
    "dandelion": {
        "name": "민들레",
        "description": "생명력이 강해 길가에서도 잘 자라며, 노란 꽃이 지고 나면 하얀 씨앗 뭉치가 바람에 날아갑니다.",
        "habitat": "들판, 길가"
    },
    "foxtail": {
        "name": "강아지풀",
        "description": "이삭의 모양이 강아지 꼬리를 닮아 붙여진 이름이며, 여름철 흔하게 볼 수 있는 한해살이풀입니다.",
        "habitat": "길가, 빈터"
    }
}

# 인덱스 번호와 영문 이름을 매칭 (모델 결과값 해석용)
class_names = ["dandelion", "foxtail", "heron", "magpie", "mallard"]