import matplotlib.pyplot as plt
import pygame
import numpy as np

# 初始化 Pygame
pygame.init()

# 设置画板大小
canvas_size = (28, 28)
window_size = (canvas_size[0] * 20, canvas_size[1] * 20)
canvas = np.zeros(canvas_size, dtype=np.uint8)

# 设置颜色
white = (255, 255, 255)
Black = (0, 0, 0)

# 初始化 Pygame 窗口
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption('Drawing Canvas')

drawing = False

# 事件循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                x, y = event.pos
                if 0 <= x < window_size[0] and 0 <= y < window_size[1]:
                    canvas[y // 20, x // 20] = 255

    # 清空屏幕并绘制画板内容
    screen.fill(Black)
    for y in range(canvas_size[1]):
        for x in range(canvas_size[0]):
            if canvas[y, x] == 255:
                pygame.draw.rect(screen, white, (x * 20, y * 20, 20, 20))

    pygame.display.flip()

saved_array = canvas
# 退出 Pygame
pygame.quit()
plt.imshow(saved_array,cmap='gray',interpolation='nearest')
plt.show()