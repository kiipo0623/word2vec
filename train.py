import numpy as np
import time
import matplotlib.pyplot as plt

class Train:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None #학습 경과 몇 iter에 한번 출력
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x) #데이터 개수
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()

        for epoch in range(max_epoch):
            #뒤섞기
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                #기울기를 구해서 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads) # 공유된 가중치를 하나로 모음
                if max_grad is not None:
                    pass #gradient clipping
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                #평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print("| epoch %d | iter %d / %d | time %d[s] | loss %.2f"
                          % (self.current_epoch+1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iter (x' + str(self.eval_interval)+')')
        plt.ylabel('loss')
        plt.show()





#매개변수 갱신할 때 매개변수의 중복을 없애는 역할
#매개변수 배열 중 중복되는 가중치를 하나로 모아
#가중치에 대응하는 기울기를 더한다

#CBOW모델 에서 input layer가 두 개고, 같은 W를 공유하고 있는 상태(self.in_layer0 == self.in_layer2)
#그대로 사용하면 안되기 때문에 중복되는 가중치를 하나로 모으고 가중치에 해당하는 기울기 더함.
def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L-1):
            for j in range(i+1, L):
                #가중치 공유 시
                if params[i] is params[j]:
                    grads[i] = grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                #가중치를 전치행렬로 공유하는 경우
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                    params[i].T.shape == params[j].shape and np.all(params[i].T == params[i]):
                    grads[i] = grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break #같은 경우가 한 번 밖에 없을거니까..?
            if find_flg: break
        if not find_flg: break
    return params, grads