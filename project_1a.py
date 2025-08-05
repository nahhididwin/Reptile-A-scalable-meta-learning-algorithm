# 3000 iterations with loss = 0.064, the original is 6000 iterations with loss = 0.074, actually I did this for high speed for adaptation, but the accuracy is only slightly lower than the original :)

# import thư viện :
import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy

# tham số (hình như là hyperparameter à, hay t nhớ nhầm?)
seed = 0 # giúp đảm bảo rằng các kết quả ngẫu nhiên có thể được tái tạo
plot = True
innerstepsize = 0.03 # stepsize in inner SGD
innerepochs = 1 # number of epochs of each inner SGD
outerstepsize0 = 0.2 # stepsize of outer optimization, i.e., meta-optimization
niterations = 10000 # number of outer updates; each iteration we sample one task and update on it

rng = np.random.RandomState(seed)
torch.manual_seed(seed)



# Tác vụ ở đây là học một hàm sine (f_randomsine ấy)
# Mỗi tác vụ được định nghĩa bằng một hàm sine với pha (phase) và biên độ (ampl) ngẫu nhiên.
# Mục tiêu của thuật toán Reptile là tìm ra một mô hình ban đầu tốt, để khi đối mặt với một hàm sine mới, nó có thể học đường cong đó một cách nhanh chóng chỉ với ntrain điểm dữ liệu ấy :)
# Define task distribution
x_all = np.linspace(-5, 5, 50)[:,None] # All of the x points
ntrain = 10 # Size of training minibatches
def gen_task():
    "Generate classification problem"
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    f_randomsine = lambda x : np.sin(x + phase) * ampl
    return f_randomsine

# Define model. Reptile paper uses ReLU, but Tanh gives slightly better results
# mạng neural network đơn giản :v, 3 layer tuyến tính, 2 layer hàm kích hoạt phi tuyến (tanh - chắc chắn ko phải là cá tanh)
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1),
)



# mấy cái function support
# totorch: Hàm chuyển đổi dữ liệu từ numpy sang tensor của PyTorch.
# train_on_batch: Thực hiện một bước SGD (Stochastic Gradient Descent - trời viết ra như vầy nhìn ngầu vl).

# Dự đoán kết quả (ypred).

# Tính loss
# backprop error loss.backward() để tính gradient.
# Cập nhật trọng số của mô hình theo gradient và innerstepsize. Đây là bước study bên trong một tác vụ.

# predict: Hàm dự đoán giá trị đầu ra cho một tập dữ liệu đầu vào.

def totorch(x):
    return ag.Variable(torch.Tensor(x))

def train_on_batch(x, y):
    x = totorch(x)
    y = totorch(y)
    model.zero_grad()
    ypred = model(x)
    loss = (ypred - y).pow(2).mean()
    loss.backward()
    for param in model.parameters():
        param.data -= innerstepsize * param.grad.data

def predict(x):
    x = totorch(x)
    return model(x).data.numpy()

# Choose a fixed task and minibatch for visualization
f_plot = gen_task()
xtrain_plot = x_all[rng.choice(len(x_all), size=ntrain)]


# for iteration in range(niterations): Vòng lặp meta-learning bên ngoài. Mỗi lần lặp, mô hình sẽ được huấn luyện trên một tác vụ ngẫu nhiên.

# weights_before = deepcopy(model.state_dict()): Lưu lại trạng thái ban đầu của mô hình trước khi bắt đầu huấn luyện trên một tác vụ mới.
# f = gen_task(): Tạo một tác vụ mới (một hàm sine với pha và biên độ ngẫu nhiên).
# Vòng lặp bên trong: Huấn luyện mô hình trên tác vụ mới này bằng cách sử dụng train_on_batch.

# weights_after = model.state_dict(): Lưu lại trạng thái của mô hình sau khi đã huấn luyện trên tác vụ mới.
# Cập nhật Reptile: Đây là điểm khác biệt của thuật toán. Thay vì cập nhật trực tiếp theo gradient, Reptile cập nhật các trọng số ban đầu của mô hình theo công thức:
# weights_initial_new = weights_initial + (weights_trained - weights_initial) * outerstepsize          (chắc thế)

# (weights_trained - weights_initial) chính là "meta-gradient". Nó cho biết mô hình đã thay đổi như thế nào sau khi học một tác vụ.
# Thuật toán sử dụng sự thay đổi này để cập nhật trạng thái ban đầu của mô hình, giúp nó trở nên "tốt hơn" để thích nghi với các tác vụ trong tương lai.


# Reptile training loop
for iteration in range(niterations):
    weights_before = deepcopy(model.state_dict())
    # Generate task
    f = gen_task()
    y_all = f(x_all)
    # Do SGD on this task
    inds = rng.permutation(len(x_all))
    for _ in range(innerepochs):
        for start in range(0, len(x_all), ntrain):
            mbinds = inds[start:start+ntrain]
            train_on_batch(x_all[mbinds], y_all[mbinds])
    # Interpolate between current weights and trained weights from this task
    # I.e. (weights_before - weights_after) is the meta-gradient
    weights_after = model.state_dict()
    outerstepsize = outerstepsize0 * (1 - iteration / niterations) # linear schedule
    model.load_state_dict({name : 
        weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize 
        for name in weights_before})



    #Phần code này dùng để vẽ đồ thị, giúp bạn thấy mô hình học nhanh như thế nào.

    # Nó lưu lại trọng số ban đầu, huấn luyện mô hình trên một tác vụ cố định trong vài bước, rồi vẽ các đường cong dự đoán sau mỗi vài bước.


    # chúng ta sẽ thấy rằng mô hình học và thích nghi với hàm sine mới một cách rất nhanh chóng sau một vài bước cập nhật. Điều này chứng tỏ thuật toán Reptile đã thành công trong việc tìm ra một mô hình ban đầu có khả năng thích nghi cao.
    
    # Periodically plot the results on a particular task and minibatch
    if plot and iteration==0 or (iteration+1) % 1000 == 0:
        plt.cla()
        f = f_plot
        weights_before = deepcopy(model.state_dict()) # save snapshot before evaluation
        plt.plot(x_all, predict(x_all), label="pred after 0", color=(0,0,1))
        for inneriter in range(32):
            train_on_batch(xtrain_plot, f(xtrain_plot))
            if (inneriter+1) % 8 == 0:
                frac = (inneriter+1) / 32
                plt.plot(x_all, predict(x_all), label="pred after %i"%(inneriter+1), color=(frac, 0, 1-frac))
        plt.plot(x_all, f(x_all), label="true", color=(0,1,0))
        lossval = np.square(predict(x_all) - f(x_all)).mean()
        plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
        plt.ylim(-4,4)
        plt.legend(loc="lower right")
        plt.pause(0.01)
        model.load_state_dict(weights_before) # restore from snapshot
        print(f"-----------------------------")
        print(f"iteration               {iteration+1}")
        print(f"loss on plotted curve   {lossval:.3f}") # would be better to average loss over a set of examples, but this is optimized for brevity



        

