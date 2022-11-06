from mpi4py import MPI
from random import randint
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
workers = comm.Get_size() - 1

mtrx1 = []
mtrx2 = []
mtrx3 = []

N = 1000


def init_matrix():
    """
        Здесь инициализируем матрицы
    """
    global mtrx1
    mtrx1 = np.random.randint(N, size=(5000, 4999))

    global mtrx2
    mtrx2 = np.random.randint(N, size=(4999, 5000))

def multiply_matrix(X, Y):
    """
    Здесь сосздадим новую матрицу, в которую запишем результат умножения
    Generate new matrix by multiplying incoming matrix data.
    Функция будет вызваться каждым дочерним def
    """
    Z = [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*Y)]
            for X_row in X]

    return Z


def distribute_matrix_data():
    def split_matrix(seq, p):
        """
        Здесь разобьем матрицы на маленькие части, которые отправятся
        в slave_operation
        """
        rows = []
        n = len(seq) / p
        r = len(seq) % p
        b, e = 0, n + min(1, r)
        for i in range(p):
            rows.append(seq[int(b):int(e)])
            r = max(0, r - 1)
            b, e = e, e + n + min(1, r)

        return rows

    rows = split_matrix(mtrx1, int(workers))

    pid = 1
    for row in rows:
        comm.send(row, dest=pid, tag=1)
        comm.send(mtrx2, dest=pid, tag=2)
        pid = pid + 1


def assemble_matrix_data():
    """
    Сборка вернет значения от slaves и создастт итоговую матрицу,
    slaves вычислят рады этой матрицы
    """
    global mtrx3

    pid = 1
    for n in range(workers):
        row = comm.recv(source=pid, tag=pid)
        mtrx3 = mtrx3 + row
        pid = pid + 1


def master_operation():
    distribute_matrix_data()
    assemble_matrix_data()


def slave_operation():
    print('получение данных от master')
    x = comm.recv(source=0, tag=1)
    y = comm.recv(source=0, tag=2)

    print('умножение матриц и отправлка результата в master')
    z = multiply_matrix(x, y)
    comm.send(z, dest=0, tag=rank)


if __name__ == '__main__':
    totalTime1 = time.time()
    if rank == 0:
        init_matrix()

        # start time
        t1 = time.time()
        print('Время начала обработки', t1)

        """for x in range(0, 9):
            x=x+1"""
        master_operation()

        # end time
        t2 = time.time()
        print('--------------------------------------------------------\n\n')
        print('Итог:\nМатрица 1:')
        print(mtrx1)

        print('Матрица 2:')
        print(mtrx2)
        print('')

        print('Матрица 3:')
        print(mtrx3)

        print('Время начала обработки:', time.ctime(t1))

        print('Время конца обработки:', time.ctime(t2))

        print('--------------------------------------------------------------')
        print('Заняло:', t2 - t1, 'сек')
        print('--------------------------------------------------------------')
        print('\n')
        totalTime2 = time.time()
        print('Итого времени:', totalTime2 - totalTime1, 'сек')
    else:
        slave_operation()
