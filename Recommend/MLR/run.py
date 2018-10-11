from mlr import mlr_test
from lr import lr_test
from plotRet import plot_ret


if __name__ == '__main__':

    m_list = [5,10,15,20]
    epoch = 2000

    for para_m in m_list:
        mlr_test(para_m,epoch)
    lr_test(epoch)
    plot_ret()