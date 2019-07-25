from simulater import Block as bl
from simulater import Space as sp
from simulater import DataManager as dm
import numpy as np
import operator
import datetime
import copy

class Environment:

    def __init__(self, filepath, width=6, height=4, name='', num_blocks=50):
        # 입력 데이터 가져오기 및 멤버변수 저장
        space, blocks = dm.import_data(filepath, width, height, num_blocks)
        self.NUM_BLOCKS = num_blocks
        self.BLOCKS = sorted(blocks, key=operator.attrgetter('_startdate'))
        self.height = space.height
        self.width = space.width
        self.space_name = space.name
        self.size = self.height * self.width
        self.LOGS = []
        self.cumulate = np.zeros([self.width, self.height])
        self.name = str(name)
        self._initialize()

    def _initialize(self):
        # episode가 시작될 때 마다 공간 정보를 초기화
        self.SPACE = sp.Space(self.width, self.height, self.space_name)
        #blocks = dm.generate_blocks(self.NUM_BLOCKS)
        #self.BLOCKS = sorted(blocks, key=operator.attrgetter('_startdate'))
        self.SPACE.update_blocks(copy.deepcopy(self.BLOCKS))
        self.STAGE = 0

    def reset(self):
        # 환경을 초기화하고 초기 상태를 반환
        self._initialize()
        status = self.SPACE.get_status(0)
        status = np.reshape(status, (1, status.size))
        return status

    def step(self, action):
        blocks = self.SPACE.get_blocks()
        num_blocks = len(blocks)
        reward = 0
        restart = False

        '''
        #실적 계획을 강제로 넣기 위한 부분
        if blocks[self.STAGE].fixed != -1:
            action = blocks[self.STAGE].fixed
        
        while self.get_state()[int(action / self.height), int(action % self.height)] != -1.0:
            action += 1
            if action == 25:
                action = 0
        '''
        x_loc = int(action / self.height)
        y_loc = int(action % self.height)

        blocks[self.STAGE].set_location(x_loc, y_loc)
        #self.SPACE.update_blocks(blocks)
        # 블록을 이동하기 전의 normalize하지 않은 상태를 저장해두고 TP이동에 활용
        #state = self.SPACE.get_status(max(0, self.STAGE - 1))
        # state를 업데이트
        is_arrangible, _ = self.SPACE.update_state_lot(self.STAGE)
        if is_arrangible:
            # InNOutSpace의 좌표가 달라서 y, x로 보내야 함
            # r = self.get_reward(state, (y_loc, x_loc))
            # r = self.transport_block(state, (y_loc, x_loc))
            r = 1
            reward += r

        self.STAGE += 1
        reward += 0
        self.LOGS.append(self.SPACE.RESULTS[-1])
        rewards = None
        if not is_arrangible:
            # self._initialize()
            reward = -1
            restart = True
            self.reset()
        elif self.STAGE == num_blocks:
            restart = True
            reward += 3
            rewards = self.substep()
            self.reset()
            # self._initialize()
        else:
            rewards = self.substep()
        status = self.get_state()
        return status, reward, restart, rewards

    # step 사이에서 반출 블록이 있는지를 체크하고 재배치 수행
    def substep(self):
        current_day = self.SPACE.event_in[self.STAGE - 1]
        next_day = datetime.datetime(datetime.MAXYEAR, 1, 1)
        if len(self.SPACE.event_in) != self.STAGE:
            next_day = self.SPACE.event_in[self.STAGE]
        if current_day == next_day:
            return
        transfers = []
        out_events = sorted(self.SPACE.event_out, key=lambda out: out[0])
        for i in range(self.STAGE):
            if current_day < out_events[i][0] <= next_day:
                transfers.append(out_events[i])
        if len(transfers) == 0:
            return
        current_blocks = []
        blocks = self.SPACE.get_blocks()
        for block in blocks:
            start, end = block.get_schedule()
            if start <= current_day < end:
                current_blocks.append(block)

        rewards = {}
        for transfer in transfers:
            state = self.SPACE.get_status(max(0, self.STAGE - 1))
            x_loc, y_loc = transfer[1].get_location()
            r = self.transport_block(state, (x_loc, y_loc), current_blocks)
            for i in range(len(blocks)):
                if transfer[1].name == blocks[i].name:
                    rewards[i] = r
        return rewards

    def get_state(self):
        state = self.SPACE.get_status(max(0, self.STAGE - 1))
        # state = self.SPACE.RESULTS[-1]
        state = self.normalize_state(state, self.STAGE)
        return state

    def normalize_state(self, state, stage):
        norm_state = np.array(state)
        blocks = self.SPACE.get_blocks()
        if len(blocks) == stage:
            stage += -1
        duration = blocks[stage].term
        for i in range(norm_state.shape[0]):
            for j in range(norm_state.shape[1]):
                if norm_state[i, j] != -1.0:
                    norm_state[i, j] = norm_state[i, j] / duration
                if norm_state[i, j] >= 3:
                    norm_state[i, j] = 3.0
        return norm_state

    def set_transporter(self, transporter):
        self.Transporter = transporter

    def get_reward(self, state, target):
        blocks = []
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] != -1.0:
                    blocks.append(np.array([j, i]))
        moves, _ = self.Transporter.get_block_moves(blocks, target)
        reward = max(0, 3 - moves)
        return reward

    def transport_block(self, state, target, current_blocks):
        blocks = []
        terms = []
        index_curr = []
        #terms.append(self.BLOCKS[self.STAGE].term)
        index = -1
        index_target = -1
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] != -1.0:
                    index += 1
                    blocks.append(np.array([i, j]))
                    terms.append(state[i, j])
                    if i == target[0] and j == target[1]:
                        index_target = index
                    for k in range(len(current_blocks)):
                        x, y = current_blocks[k].get_location()
                        if x == i and y == j:
                            index_curr.append(k)
        if index_target == -1:
            print('here')
        moves, moved_blocks = self.Transporter.get_block_moves(blocks, index_target, self.name)
        #moved_blocks.append(moved_blocks.pop(0))
        moved_state = np.full(state.shape, -1.0, dtype=float)
        try:
            for i in range(len(moved_blocks)):
                if i == index_target:
                    continue
                current_blocks[index_curr[i]].set_location(moved_blocks[i][0], moved_blocks[i][1])
                moved_state[moved_blocks[i][0], moved_blocks[i][1]] = terms[i] - terms[index_target]

            self.SPACE.modify_latest_state(moved_state)
            self.LOGS.append(self.SPACE.RESULTS[-1])
            del current_blocks[index_curr[index_target]]
        except:
            print('here')
        if moves == 0:
            reward = 2
        else:
            reward = max(0, 1/moves)
        print(moves)
        return reward
