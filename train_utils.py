from math import pi
import numpy as np
import random

from config import args
def test_model(model, mode='static', display=True):
    # i = 0
    # test_game = Gridworld(mode=mode)
    # state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
    # state = torch.from_numpy(state_).float()
    # if display:
    #     print("Initial State:")
    #     print(test_game.display())
    # status = 1
    # while(status == 1): #A
    #     qval = model(state)
    #     qval_ = qval.data.numpy()
    #     action_ = np.argmax(qval_) #B
    #     action = action_set[action_]
    #     if display:
    #         print('Move #: %s; Taking action: %s' % (i, action))
    #     test_game.makeMove(action)
    #     state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
    #     state = torch.from_numpy(state_).float()
    #     if display:
    #         print(test_game.display())
    #     reward = test_game.reward()
    #     if reward != -1:
    #         if reward > 0:
    #             status = 2
    #             if display:
    #                 print("Game won! Reward: %s" % (reward,))
    #         else:
    #             status = 0
    #             if display:
    #                 print("Game LOST. Reward: %s" % (reward,))
    #     i += 1
    #     if (i > 15):
    #         if display:
    #             print("Game lost; too many moves.")
    #         break
    
    # win = True if status == 2 else False
    # return win
    pass

# def print_accuracy(model):
#     max_games = 1000
#     wins = 0
#     for i in range(max_games):
#         win = test_model(model, mode='random', display=False)
#         if win:
#             wins += 1
#     win_perc = float(wins) / float(max_games)
#     print("Games played: {0}, # of wins: {1}".format(max_games,wins))
#     print("Win percentage: {}%".format(100.0*win_perc))

def clacDuration(yaw_rate, radian):
    duration = (pi / radian) / (yaw_rate * pi / 180)
    return duration

def calcValues(qval, eps):
    action_vector = qval[:3]
    one_hot = np.zeros_like(action_vector)
    max_value = np.argmax(action_vector)
    radian = qval[3]

    if random.random() < eps:
        max_value = np.random.randint(0, args.n_classes - 1)
        radian = random.uniform(-1, 1)
        
    one_hot[max_value] = 1
    action_vector = one_hot * args["drone"].moving_unit

    if radian < -1:  radian %= -1
    elif radian > 1: radian %= 1

    return *action_vector, radian

def calcReward(map_pcd, prev_pcd, curr_pcd, client):

    # 상태에서 필요한 정보 추출
    collision_info = client.collision_info

    # 보상 초기값
    reward = 0.0
    
    # 만약 드론이 충돌한 경우
    if collision_info.has_collided:
        reward = -10.0  # 큰 음수 값으로 보상

    # 이전 포인트 클라우드와 현재 포인트 클라우드의 차이 계산
    map_np = np.asarray(map_pcd.points)
    prev_np = np.asarray(prev_pcd.points)
    curr_np = np.asarray(curr_pcd.points)
    
    # 두 포인트 클라우드의 크기를 동일하게 맞춤
    min_points_mc = min(map_np.shape[0], curr_np.shape[0])
    mse_mc = np.mean((map_np[:min_points_mc] - curr_np[:min_points_mc]) ** 2)
    min_points_pc = min(prev_np.shape[0], curr_np.shape[0])
    mse_pc = np.mean((prev_np[:min_points_pc] - curr_np[:min_points_pc]) ** 2)
    
    # 전체 맵과 현재 만든 맵의 차이가 크면 작은 음의 보상
    # 계속해서 진행하도록
    if mse_mc > args.voxel_threshold:
        reward -= 1.0
    # 전체 맵과 현재 만든 맵의 차이가 작으면 큰 양의 보상
    # 전체 맵을 잘 만든 것이므로
    elif mse_mc < args.voxel_threshold:
        reward = 10.0
    # 전에 만든 맵과 현재 만든 맵의 차이가 크면 작은 음의 보상
    # 계속해서 진행하도록
    elif mse_pc > args.voxel_threshold:
        reward -= 1.0
    # 전에 만든 맵과 현재 만든 맵의 차이가 작으면 큰 음의 보상
    # 드론이 조금 움직인 것이므로 
    elif mse_pc < args.voxel_threshold:
        reward -= 10.0

    return reward