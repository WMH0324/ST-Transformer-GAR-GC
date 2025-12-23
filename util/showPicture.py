import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
import matplotlib.colors as mcolors
from .collective import FRAMES_SIZE, ACTIONS, ACTIVITIES


# 不同长度数据，统一为一个标准，倍乘x轴
def multiple_equal(x, y):
    x_len = len(x)
    y_len = len(y)
    times = x_len / y_len
    y_times = [i * times for i in y]
    return y_times


# ------------用于显示训练和测试中的损失值变化------------
def showLoss(train_loss, test_loss):
    y_train_loss = train_loss  # 训练损失值，即y轴
    x_train_loss = range(1, len(y_train_loss) + 1)  # 训练次数，即x轴
    y_test_loss = test_loss  # 测试损失值，即y轴
    x_test_loss = range(1, len(y_test_loss) + 1)  # 测试次数，即x轴
    print()
    plt.figure()
    plt.grid(linestyle='--')  # 生成网格

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epochs', size=18)  # x轴标签
    plt.ylabel('loss', size=18)  # y轴标签

    # 设置坐标轴的显示范围
    plt.xlim(0, len(y_train_loss))
    plt.ylim(0, 5)

    # 设置坐标轴刻度
    x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)

    # 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 增加参数color='red',这是红色。
    # 训练曲线
    plt.plot(x_train_loss, y_train_loss, color='red', linewidth=1, linestyle="solid", label="train loss")
    plt.scatter(x_train_loss, y_train_loss, color='red')
    # for i in range(len(x_train_loss)):
    #     plt.text(x_train_loss[i],y_train_loss[i]+0.1,"%.2f"%y_train_loss[i])

    # 测试曲线
    plt.plot(x_test_loss, y_test_loss, color='blue', linewidth=1, linestyle="solid", label="test loss")
    plt.scatter(x_test_loss, y_test_loss, color='blue')
    # for i in range(len(x_test_loss)):
    #     plt.text(x_test_loss[i],y_test_loss[i]-0.3,"%.2f"%y_test_loss[i])
    plt.rcParams.update({'font.size': 14})
    plt.legend()
    plt.title('Loss curve', size=20)
    plt.savefig('/kaggle/working/train_and_test_loss.jpg')
    plt.show()


# ------------用于显示训练和测试中的准确率变化------------
def showAccuracy(train_accuracy, test_accuracy):
    print(train_accuracy, test_accuracy)
    y_train_acc = train_accuracy  # 训练准确率值，即y轴
    x_train_acc = range(1, len(y_train_acc) + 1)  # 训练次数，即x轴
    y_test_acc = test_accuracy  # 测试准确率值，即y轴
    x_test_acc = range(1, len(y_test_acc) + 1)  # 测试次数，即x轴
    plt.figure()

    plt.grid(linestyle='--')  # 生成网格

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置坐标轴的显示范围
    plt.xlim(0, len(y_train_acc))
    plt.ylim(30, 100)

    # 设置坐标轴刻度
    x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(5)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlabel('epochs', size=18)  # x轴标签
    plt.ylabel('accuracy', size=18)  # y轴标签

    # 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 增加参数color='red',这是红色。
    # 训练曲线
    plt.plot(x_train_acc, y_train_acc, color='red', linewidth=1, linestyle="solid", label="train accuracy")
    plt.scatter(x_train_acc, y_train_acc, color='red')
    # for i in range(len(x_train_acc)):
    #     plt.text(x_train_acc[i],y_train_acc[i]-10,"%.2f"%y_train_acc[i]+"%")

    # 测试曲线
    plt.plot(x_test_acc, y_test_acc, color='blue', linewidth=1, linestyle="solid", label="test accuracy")
    plt.scatter(x_test_acc, y_test_acc, color='blue')
    for i in range(len(x_test_acc)):
        plt.text(x_test_acc[i],y_test_acc[i]+10,"%.2f"%y_test_acc[i]+"%")
    plt.rcParams.update({'font.size': 14})
    plt.legend()
    plt.title('Accuracy curve', size=20)
    plt.savefig('/kaggle/working/train_and_test_accuracy.jpg')
    plt.show()


# ------------用于显示训练各类的准确率变化------------
def showTrainPerAccuracy(train_per_accuracy_total):
    plt.figure()
    plt.grid(linestyle='--')  # 生成网格
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 设置坐标轴的显示范围
    plt.xlim(0, 20)
    plt.ylim(-1, 101)

    # 设置坐标轴刻度
    x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(10)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlabel('epochs', size=18)  # x轴标签
    plt.ylabel('accuracy', size=18)  # y轴标签

    colors = {i: a for i, a in enumerate(mcolors.TABLEAU_COLORS.keys())}

    for i, train_per_accuracy in enumerate(train_per_accuracy_total):
        y_train_acc = train_per_accuracy  # 训练准确率值，即y轴
        x_train_acc = range(1, len(y_train_acc) + 1)  # 训练次数，即x轴
        # y_test_acc = test_accuracy              # 测试准确率值，即y轴
        # x_test_acc = range(1,len(y_test_acc)+1)    # 测试次数，即x轴

        # 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 增加参数color='red',这是红色。
        # 训练曲线

        plt.plot(x_train_acc, y_train_acc, color=colors[i], linewidth=1, linestyle="solid", label=ACTIVITIES[i])
        plt.scatter(x_train_acc, y_train_acc, color=colors[i])
        # for i in range(len(x_train_acc)):
        #     plt.text(x_train_acc[i],y_train_acc[i]-10,"%.2f"%y_train_acc[i]+"%")

    # for i in range(len(x_test_acc)):
    #     plt.text(x_test_acc[i],y_test_acc[i]+10,"%.2f"%y_test_acc[i]+"%")
    plt.rcParams.update({'font.size': 14})
    plt.legend()
    plt.title('Accuracy curve for each category', size=20)
    plt.savefig('/kaggle/working/Train_per_accuracy.jpg')
    plt.show()


# ------------用于显示测试中各类的准确率变化------------
def showTestPerAccuracy(test_per_accuracy_total):
    plt.figure()
    plt.grid(linestyle='--')  # 生成网格
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 设置坐标轴的显示范围
    plt.xlim(0, 20)
    plt.ylim(-1, 101)

    # 设置坐标轴刻度
    x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(10)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlabel('epochs', size=18)  # x轴标签
    plt.ylabel('accuracy', size=18)  # y轴标签

    colors = {i: a for i, a in enumerate(
        mcolors.TABLEAU_COLORS.keys())}

    for i, test_per_accuracy in enumerate(test_per_accuracy_total):
        y_test_acc = test_per_accuracy  # 训练准确率值，即y轴
        x_test_acc = range(1, len(y_test_acc) + 1)  # 训练次数，即x轴
        # y_test_acc = test_accuracy              # 测试准确率值，即y轴
        # x_test_acc = range(1,len(y_test_acc)+1)    # 测试次数，即x轴

        # 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 增加参数color='red',这是红色。
        # 训练曲线

        plt.plot(x_test_acc, y_test_acc, color=colors[i], linewidth=1, linestyle="solid", label=ACTIVITIES[i])
        plt.scatter(x_test_acc, y_test_acc, color=colors[i])
        # for i in range(len(x_train_acc)):
        #     plt.text(x_train_acc[i],y_train_acc[i]-10,"%.2f"%y_train_acc[i]+"%")

    # for i in range(len(x_test_acc)):
    #     plt.text(x_test_acc[i],y_test_acc[i]+10,"%.2f"%y_test_acc[i]+"%")
    plt.rcParams.update({'font.size': 14})
    plt.legend()
    plt.title('Accuracy curve for each category', size=20)
    plt.savefig('Test_per_accuracy.jpg')
    plt.show()
