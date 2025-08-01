import zmq
from Sparrow_V2 import Sparrow_PlayGround, str2bool
from utils.Transqer import Transqer_agent
import torch, pygame
import argparse

# server.py
#from sockets import init_sockets, global_socket_client1

parser = argparse.ArgumentParser()
'''Hyperparameter Setting for Transqer'''
parser.add_argument('--ModelIdex', type=int, default=2450, help='which model(e.g. 2450k.pth) to load')
parser.add_argument('--net_width', type=int, default=64, help='Linear net width')
parser.add_argument('--T', type=int, default=10, help='length of time window')
parser.add_argument('--H', type=int, default=8, help='Number of Head')
parser.add_argument('--L', type=int, default=3, help='Number of Transformer Encoder Layers')

'''Hyperparameter Setting for Sparrow'''
parser.add_argument('--dvc', type=str, default='cuda', help='running device of Sparrow: cuda / cpu')
parser.add_argument('--action_type', type=str, default='Discrete', help='Action type: Discrete / Continuous')
parser.add_argument('--window_size', type=int, default=800, help='size of the map')
parser.add_argument('--D', type=int, default=400, help='maximal local planning distance:366*1.414')
parser.add_argument('--N', type=int, default=1, help='number of vectorized environments')
parser.add_argument('--O', type=int, default=15, help='number of obstacles in each environment')
parser.add_argument('--RdON', type=str2bool, default=False, help='whether to randomize the Number of dynamic obstacles')
parser.add_argument('--ScOV', type=str2bool, default=False, help='whether to scale the maximal velocity of dynamic obstacles')
parser.add_argument('--RdOV', type=str2bool, default=True, help='whether to randomize the Velocity of dynamic obstacles')
parser.add_argument('--RdOT', type=str2bool, default=True, help='whether to randomize the Type of dynamic obstacles')
parser.add_argument('--RdOR', type=str2bool, default=True, help='whether to randomize the Radius of obstacles')
parser.add_argument('--Obs_R', type=int, default=14, help='maximal obstacle radius, cm')
parser.add_argument('--Obs_V', type=int, default=30, help='maximal obstacle velocity, cm/s')
parser.add_argument('--MapObs', type=str, default=None, help="name of map file, e.g. 'map.png' or None")
parser.add_argument('--ld_a_range', type=int, default=360, help='max scanning angle of lidar (degree)')
parser.add_argument('--ld_d_range', type=int, default=300, help='max scanning distance of lidar (cm)')
parser.add_argument('--ld_num', type=int, default=72, help='number of lidar streams in each world')
parser.add_argument('--ld_GN', type=int, default=3, help='how many lidar streams are grouped for one group')
parser.add_argument('--ri', type=int, default=0, help='render index: the index of world that be rendered')
parser.add_argument('--basic_ctrl_interval', type=float, default=0.1, help='control interval (s), 0.1 means 10 Hz control frequency')
parser.add_argument('--ctrl_delay', type=int, default=0, help='control delay, in basic_ctrl_interval, 0 means no control delay')
parser.add_argument('--K', type=tuple, default=(0.55,0.6), help='K_linear, K_angular')
parser.add_argument('--draw_auxiliary', type=str2bool, default=False, help='whether to draw auxiliary infos')
parser.add_argument('--render_speed', type=str, default='fast', help='fast / slow / real')
parser.add_argument('--max_ep_steps', type=int, default=500, help='maximum episodic steps')
parser.add_argument('--noise', type=str2bool, default=True, help='whether to add noise to the observations')
parser.add_argument('--DR', type=str2bool, default=True, help='whether to use Domain Randomization')
parser.add_argument('--DR_freq', type=int, default=int(3.2e3), help='frequency of Domain Randomization, in total steps')
parser.add_argument('--compile', type=str2bool, default=False, help='whether to use torch.compile to boost simulation speed')
opt = parser.parse_args()
opt.render_mode = None
opt.dvc = torch.device(opt.dvc)

envs = Sparrow_PlayGround(**vars(opt))
opt.state_dim = envs.state_dim
opt.action_dim = envs.action_dim
print("\nPress 'Esc' to reset environment.")

# Init Transqer agent
agent = Transqer_agent(opt)
agent.load(opt.ModelIdex)

agent.queue.clear()
s, info = envs.reset()

#加代码，一个function()
def test(relative_real_x_color, relative_real_y_color):

    return relative_real_x_color, relative_real_y_color

#ipconfig 查看IPv4地址
#netsh advfirewall set allprofiles state off 临时关闭防火墙
context = zmq.Context()

socket_client1 = context.socket(zmq.REP)
socket_client1.bind("tcp://0.0.0.0:0226")

socket_client2 = context.socket(zmq.REP)
socket_client2.bind("tcp://0.0.0.0:0227")

#netstat -ano | findstr 0225  用于查找是否有这个监听窗口
#init_sockets()

a_space = torch.tensor([[0.2*envs.v_linear_max , envs.v_angular_max],  # 动作 0：低速前进 + 右转
                                     [envs.v_linear_max , envs.v_angular_max], # 动作 1：全速前进 + 右转
                                     [envs.v_linear_max, 0], # 动作 2：全速前进
                                     [envs.v_linear_max, - envs.v_angular_max],     # 动作 3：全速前进 + 左转
                                     [0.2*envs.v_linear_max, - envs.v_angular_max], # 动作 4：低速前进 + 左转
                                     [-envs.v_linear_max, 0],  # 动作 5：全速后退
                                     [0.1*envs.v_linear_max, 0], # 动作 6：减速
                                     [0., 0.]], device = envs.dvc) # 动作 7：停止

a_space_numpy = a_space.cpu().numpy()

# 创建轮询器同时监听两个套接字
poller = zmq.Poller()
poller.register(socket_client1, zmq.POLLIN)
poller.register(socket_client2, zmq.POLLIN)

while True:
    # reset() env 渲染
    '''keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        agent.queue.clear()
        s, info = envs.reset()'''
    
    # 轮询套接字（非阻塞）
    socks = dict(poller.poll())

    
    agent.queue.append(s)  # 将s加入时序窗口队列
    TW_s = agent.queue.get()  # 取出队列所有数据及

    if envs.win_vec:
        a = 7 * torch.ones((opt.N,), dtype=torch.long,
                           device=opt.dvc)  # 在a_space中最后一个动作选择是停止。 在离散动作空间中，a是a_space的索引，不是action本身。
    else:
        a = agent.select_action(TW_s, deterministic=False)

    action_index = a.item()

    action_Discrete = a_space_numpy[action_index]
    print("action_Discrete: ", action_Discrete)

    # 通讯
    if socket_client1 in socks:
        message1 = socket_client1.recv_json()
        print("Received message1 begin work:", message1)  # Debugging: Print the received message
        try:
            xx, yy, zz = message1['xx'], message1['yy'], message1['zz']
            #c1, c2 = test(relative_real_x_color, relative_real_y_color)
            # c=a+b #调用函数
            data_to_send1 = {
                'c1': float(action_Discrete[0]),
                'c2': float(action_Discrete[1]),
                't2': zz
            }
            socket_client1.send_json(data_to_send1)
            print("send message1 !")

        except KeyError as e:
            print(f"KeyError: {e}. Message received: {message1}")
            socket_client1.send_json({'error': str(e), 'message': message1})
        


    #print("in the step, a is: ", a)

    # 加一个发出消息的try,执行动作
    # 通讯
    if socket_client2 in socks:
        message2 = socket_client2.recv_json()
        print("Received message2 begin work:", message2)  # Debugging: Print the received message
        try:
            relative_real_x_color, relative_real_y_color = message2['relative_real_x_color'], message2['relative_real_y_color']
            c11, c22 = test(relative_real_x_color, relative_real_y_color)
            #t22 = t
            # c=a+b #调用函数
            data_to_send2 = {
                'c11': c11,
                'c22': c22
            }
            socket_client2.send_json(data_to_send2)
            print("socket2 send message! ")
        except KeyError as e:
            print(f"KeyError: {e}. Message received: {message2}")
            socket_client2.send_json({'error': str(e), 'message': message2})
        
        print("relative_real_x_color: ", relative_real_x_color)
        print("relative_real_y_color: ", relative_real_y_color)
    #? 把relative_real_x_color, relative_real_y_color改成和opencv_color的共享变量
    #! 用点亮观察labview中的运行顺序

    #print('111111')
    #下策? 等robo靠近下一个目标位置时再更新整个环境包括渲染??? 看看怎么把labview中的判断条件接进去?
        s, r, dw, tr, info = envs.step(a, relative_real_x_color, relative_real_y_color) #step里一个接收消息，直接修正s
   