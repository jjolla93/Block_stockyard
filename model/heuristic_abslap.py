from simulater import DataManager
from model.trained_transporter import Transporter
import operator
import numpy as np
import tensorflow as tf

class HeuristicABSLAP(object):
    def __init__(self, path, width=5, height=5):
        # 입력 데이터 가져오기 및 멤버변수 저장
        space, blocks = DataManager.import_data_from_csv(path, width, height)
        self.num_block = len(blocks)
        self.blocks = sorted(blocks, key=operator.attrgetter('_startdate'))
        self.height = space.height
        self.width = space.width
        self.rearrange = 0
        sess = tf.Session()
        self.transporter = Transporter(sess, width, height, mode=0)

    def arrange_blocks(self):
        yard = np.full([self.height, self.width], 0) #적치장의 각 지번 0으로 초기화 함. 여기에 값 채워서 계산
        current_date = None
        #블록을 하나씩 투입, 후보 지번들을 찾고 비용이 최소인 지번을 선택
        for block in self.blocks:
            schedule = block.get_schedule()
            if current_date:
                days = (schedule[0] - current_date).days
                #다음 블록 투입시에 적치된 블록들의 잔여일을 업데이트
                self.update_state(yard, days)
            current_date = schedule[0]
            candidates = self.get_candidates(yard)
            candidate = self.choose_candidate(yard, candidates)
            yard[candidate // yard.shape[1], candidate % yard.shape[1]] = block.term
            print(yard)
        #블록 투입이 끝난 이후 남은 블록들을 하나씩 반출
        while np.sum(yard) != 0:
            self.update_state(yard, np.min(yard[np.nonzero(yard)]))

        positions = []  # 결정된 각 블록의 배치 지번을 리스트로 출력
        return positions

    def get_candidates(self, state, exception=[]):
        candidates = []
        for i, row in enumerate(state):
            if sum(row) == 0:
                #candidates.append(state.shape[1] * i + state.shape[0] // 2)
                candidates.append(state.shape[1] * i)
            for j, val in enumerate(row):
                if j == 0:
                    continue
                if (i, j) in exception:
                    continue
                if val == 0 and row[j - 1] != 0:
                    candidates.append(state.shape[1] * i + j)
        #이웃하는 후보 지번이 두 개 이상 있으면 후보에서 제거
        num = len(candidates)
        for i, candidate in enumerate(candidates[::-1]):
            adjacent = 0
            if candidate - 1 in candidates:
                adjacent += 1
            if candidate + 1 in candidates:
                adjacent += 1
            if candidate - state.shape[1] in candidates:
                adjacent += 1
            if candidate + state.shape[1] in candidates:
                adjacent += 1
            if adjacent >= 2:
                del candidates[num - i - 1]
        return candidates

    def update_state(self, state, days):
        outbounds = []
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == 0:
                    continue
                elif state[i, j] > days:
                    state[i, j] -= days
                else:
                    #잔여일이 0이하가 되는 블록들은 반출 블록 리스트에 추가
                    outbounds.append((i, j))
                    state[i, j] = 0
        for outbound in outbounds:
            self.checked = []
            self.checked.append(outbound)
            #outbound 블록이 반출이 가능한지를 확인하고 불가능한 경우에 재배치 수행
            #if not self.check_outbound(state, outbound):
            #    self.rearrange_block(state, outbound)
        #Transporting agent를 활용하여 재배치 수행
        if outbounds:
            self.rearrange_by_transporter(state, outbounds)

    def choose_candidate(self, state, candidates, block=None):
        cost = float('inf')
        min_candidate = -1
        for candidate in candidates:
            if block:
                _cost = abs(block[0] - candidate // state.shape[1]) + abs(block[1] - candidate % state.shape[1])
            else:
                _cost = (candidate - 1) % state.shape[1]
            if _cost < cost:
                min_candidate = candidate
                cost = _cost
        return min_candidate

    def rearrange_block(self, state, block):
        target_i, target_j = block[0], block[1]
        print(state)
        for i in range(1, state.shape[1] - target_j):
            if state[target_i, target_j + i] != 0:
                candidates = self.get_candidates(state, [(target_i, target_j + i + 1)])
                candidate = self.choose_candidate(state, candidates, block)
                state[candidate // state.shape[1], candidate % state.shape[1]] = state[target_i, target_j + i]
                state[target_i, target_j + i] = 0
                self.rearrange += 1
                print('rearrange: {0}'.format(self.rearrange))
                print(state)

    def rearrange_by_transporter(self, state, outbounds):
        blocks = []
        out_indices = []
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] != 0:
                    blocks.append((j, i))
        for outbound in outbounds:
            out_indices.append(len(blocks))
            blocks.append((outbound[1], outbound[0]))
        blocks_clone = blocks[:]
        for index in out_indices:
            moves, moved_blocks = self.transporter.get_block_moves(blocks, index, 0)
            blocks = moved_blocks
            self.rearrange += moves
            print('moves: {0}'.format(self.rearrange))
        new_state = np.full(state.shape, 0)
        for i, block in enumerate(blocks_clone):
            new_state[blocks[i][1], blocks[i][0]] = state[block[1], block[0]]
        state[:] = new_state

    def check_outbound(self, state, block):
        if block[1] == state.shape[1] - 1:
            return True
        directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
        for i in range(4):
            search = (block[0] + directions[i][0], block[1] + directions[i][1])
            if 0 <= search[0] < state.shape[0] and 0 <= search[1] < state.shape[1]:
                if state[search[0], search[1]] != 0 or search in self.checked:
                    continue
                else:
                    self.checked.append(search)
                    if self.check_outbound(state, search):
                        return True
        return False


if __name__ == "__main__":
    abslap = HeuristicABSLAP('../data/data.csv')
    arranged = abslap.arrange_blocks()
    print(arranged)