import numpy as np
import timeit


def GenerateIslandMatrix(size):
	result = []
	for i in range(size):
		result.append(np.random.choice([0, 1], size=size, p=[0.4, 0.6]))
	result = np.asarray(result)
	return result


def numIslands(matrix):
	islandCount = 0
	visited = []
	islandQueue = []
	for i in matrix:
		for j in i:
			if (i, j) in visited:
				continue
			if matrix[i, j] == 0:
				continue
			islandQueue.append((i, j))
			while len(islandQueue) > 0:
				visited.append(islandQueue.pop(0))
				# Get neighbors
				up = (i - 1, j)
				if up[0] < 0 or up[1] < 0 or up[0] > matrix.shape[0] or up[1] > matrix.shape[0]:
					continue
				elif matrix[up] == 0:
					continue
				else:
					islandQueue.append(up)

				down = (i + 1, j)
				if down[0] < 0 or down[1] < 0 or down[0] > matrix.shape[0] or down[1] > matrix.shape[0]:
					continue
				elif matrix[down] == 0:
					continue
				else:
					islandQueue.append(down)

				left = (i, j - 1)
				if left[0] < 0 or left[1] < 0 or left[0] > matrix.shape[0] or left[1] > matrix.shape[0]:
					continue
				elif matrix[left] == 0:
					continue
				else:
					islandQueue.append(left)

				right = (i, j + 1)
				if right[0] < 0 or right[1] < 0 or right[0] > matrix.shape[0] or right[1] > matrix.shape[0]:
					continue
				elif matrix[right] == 0:
					continue
				else:
					islandQueue.append(right)

			islandCount += 1



if __name__ == "__main__":
	print("hey")
	print()
	np.random.seed(5)

	matrix_1 = GenerateIslandMatrix(5)
	matrix_2 = GenerateIslandMatrix(7)
	matrix_3 = GenerateIslandMatrix(9)
	matrix_4 = GenerateIslandMatrix(11)
	matrix_5 = GenerateIslandMatrix(15)
	matrix_6 = GenerateIslandMatrix(25)
	matrix_7 = GenerateIslandMatrix(50)
	matrix_8 = GenerateIslandMatrix(100)
	matrix_9 = GenerateIslandMatrix(500)
	print()
