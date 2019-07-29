from simulater import DataManager
import operator
import numpy as np

class HeuristicABSLAP(object):
    def __init__(self, path, width=5, height=5):
        # 입력 데이터 가져오기 및 멤버변수 저장
        space, blocks = DataManager.import_data_from_csv(path, width, height)
        self.num_block = len(blocks)
        self.blocks = sorted(blocks, key=operator.attrgetter('_startdate'))
        self.height = space.height
        self.width = space.width

    def arrange_blocks(self):
        yard = np.full([self.height, self.width], 0) #적치장의 각 지번 0으로 초기화 함. 여기에 값 채워서 계산
        current_date = None
        for block in self.blocks:
            schedule = block.get_schedule()
            if current_date:
                days = (schedule[0] - current_date).days
                self.update_state(yard, days)
            current_date = schedule[0]
            candidates = self.get_candidates(yard)
            candidate = self.choose_candidate(yard, candidates)
            print(candidate)
            yard[candidate // yard.shape[1], candidate % yard.shape[1]] = block.term
            print(yard)
        positions = []  # 결정된 각 블록의 배치 지번을 리스트로 출력
        return positions

    def get_candidates(self, state):
        candidates = []
        for i, row in enumerate(state):
            if sum(row) == 0:
                candidates.append(state.shape[1] * i + state.shape[0] // 2)
            for j, val in enumerate(row):
                if j == 0:
                    continue
                if val == 0 and row[j - 1] != 0:
                    candidates.append(state.shape[1] * i + j)
        return candidates

    def update_state(self, state, days):
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == 0:
                    continue
                elif state[i, j] > days:
                    state[i, j] -= days
                else:
                    state[i, j] = 0

    def choose_candidate(self, state, candidates):
        cost = float('inf')
        min_candidate = -1
        for candidate in candidates:
            _cost = state[candidate // state.shape[1], candidate % state.shape[1]]
            if _cost < cost:
                min_candidate = candidate
        return min_candidate



if __name__ == "__main__":
    abslap = HeuristicABSLAP('../data/test_data3.csv')
    arranged = abslap.arrange_blocks()
    print(arranged)