from simulater import DataManager
import operator
import numpy as np


class TabuSearch(object):
    def __init__(self, path, width=5, height=5):
        # 입력 데이터 가져오기 및 멤버변수 저장
        space, blocks = DataManager.import_data_from_csv(path, width, height)
        self.num_block = len(blocks)
        #block 클래스에 각 블록의 입고일과 출고일이 datetime으로 들어 있음
        self.blocks = sorted(blocks, key=operator.attrgetter('_startdate'))
        self.height = space.height
        self.width = space.width

    def arrange_blocks(self):
        yard = np.full([self.height, self.width], 0) #적치장의 각 지번 0으로 초기화 함. 여기에 값 채워서 계산
        positions = []  # 결정된 각 블록의 배치 지번을 리스트로 출력
        return positions


if __name__ == "__main__":
    tabu = TabuSearch('../data/test_data3.csv')
    arranged = tabu.arrange_blocks()
    print(arranged)