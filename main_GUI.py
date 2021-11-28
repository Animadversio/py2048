import pygame as pg
pg.display.set_caption("SLG board")
SCREEN = pg.display.set_mode((400, 500)) # get screen
screen = pg.display.get_surface()
# Setup Font
pg.font.init()
font = pg.font.SysFont("microsoftsansserif", 20, bold=False)
board_WIDTH, board_HEIGHT = 400, 500
TileW = 100
Margin = 5
NUMCOLOR = {NULLVAL: (200, 200, 200), 1: (190, 190, 200), 2: (180, 180, 200), 4: (180, 180, 210), 8: (170, 170, 210),
            16: (165, 165, 220), 32: (155, 155, 220), 64: (145, 145, 225), 128: (135, 135, 225), 256: (125, 125, 230),
            512: (110, 110, 240), 1024: (95, 95, 240), 2048: (80, 80, 250)}

def drawBoard(board, score):
    pg.draw.rect(screen, (240, 240, 240), pg.Rect(0, 0, board_WIDTH, board_HEIGHT), 0)
    for i in range(dimen):
        for j in range(dimen):
            clr = NUMCOLOR[board[i, j]]
            pg.draw.rect(screen, clr,
                         pg.Rect(i * TileW + Margin, j * TileW + Margin, TileW - 2 * Margin, TileW - 2 * Margin))
            if board[i, j] != NULLVAL:
                img = font.render('%d' % board[i, j], True, (0,0,0))
                screen.blit(img, ((i + 1/2) * TileW - 28, (j + 1/2) * TileW - 18))
    scoreimg = font.render('%d' % score, True, (100, 40, 40))
    screen.blit(scoreimg, (200, 450))

def GUI_loop(board=None, score=0):
    if board is None:
        board = getInitState()
        score = 0
    exitFlag = False
    actseq = []
    while not exitFlag:
        act = None
        for event in pg.event.get():
            if event.type == pg.QUIT:
                exitFlag = True
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP: act = LEFT
                if event.key == pg.K_DOWN: act = RIGHT
                if event.key == pg.K_LEFT: act = UP
                if event.key == pg.K_RIGHT: act = DOWN
        if act is not None:
            actseq.append(act)
            board, reward, finished = getSuccessor(board, action=act, show=False)
            score += reward
            if finished:
                exitFlag = True
        drawBoard(board, score)
        pg.display.update()
