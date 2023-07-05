import socket, sys
import math
import random


interactive_flag = False

depth_analysis = 3

def pos2_to_pos1(x2):
    return x2[0] * 8 + x2[1]


def pos1_to_pos2(x):
    row = x // 8
    col = x % 8
    return [row, col]


# Função objetivo! (Nota: Reaproveitando parte da função objetivo do professor.)
# Irei fazer a minha função objetivo com base no algoritmo alpha-beta prunning, porque analisará apenas as jogadas que são necessárias até chegarmos à vitória e por consequência este algoritmo é mais rápido de executar.
# Esta função objetivo tem como base a contabilização da posições e do número de peças no tabuleiro, conforme a avaliação no posicionamento das mesmas a minha probabilidade de ganhar é quase certa.
# Também considero o Simulated Annealing onde faço uma busca local e onde verifico se a posição de cada peça em específico irá ser atacada quer na diagonal ou na horizontal, com base nesta verificação o meu algorítmo (função objetivo) avalia para quais posições as mesmas podem ir de acordo com as respetivas movimentações que possam fazer.
def f_obj(board, play):

    # Declaro as minhas peças no tabuleiro inicial
    brancas = 'abcdeghijklmnop'
    negras = 'ABCDEFGHIJKLMNOP'

    # Pontuação de cada peça (testar contra a função objetivo do professor)
    pontos = [10, 7, 6, 100, 9999, 6, 7, 10, 1, 1, 1, 1, 1, 1, 1, 1]

    # Caso jogue pelo lado das brancas (primeira movimentação a ser feita é deste lado):
    # Pontuação
    pontos_brancas = 0
    # Peças brancas não atacadas
    nao_atacadas_brancas = 0 

    # Percorro todas as peças brancas
    for i, pecas_brancas in enumerate(brancas):
        # Procuro se o tabuleiro contém todas as peças brancas
        estado_b = board.find(pecas_brancas)
        # Caso tenha todas as peças
        if estado_b >= 0:
            # Atribuição dos pontos correspondentes a cada peça branca
            pontos_brancas += pontos[i]
            # Irá descrever o movimento da peça no tabuleiro
            posicao_b = pos1_to_pos2(estado_b)

    
    # Caso jogue do lado das negras
    # Pontuação
    pontos_negras = 0
    # Peças negras não atacadas
    nao_atacadas_negras = 0

    # Percorro todas as peças negras
    for i, pecas_negras in enumerate(negras):
        # Procuro se o tabuleiro contém todas as peças brancas
        estado_n = board.find(pecas_negras)
        # Caso tenha todas as peças
        if estado_n >= 0:
            # Atribuição dos pontos correspondentes a cada peça branca
            pontos_negras += pontos[i]
            # Irá descrever o movimento da peça no tabuleiro
            posicao_n = pos1_to_pos2(estado_n)

    # Percorre todo o tabuleiro, retirando as peças negras e certifica-se que nenhuma branca é atacada
    for l in range((len(board) - 1) - (estado_b + estado_n)):
        nao_atacadas_brancas = 0
        for c in range(l + 1, len(board) - (estado_b + estado_n)):

            # Verifica se há algum ataque ou na diagonal ou na horizontal           

            # Processo iterativo para cada coluna e linha
            if (c != l) and (c != l + (c - l)) and (c != l - (c - l)):

                # Se não houver nenhum ataque na coluna que está a ser iterada a pontuação é aumentada naquela posição, esta variável é apenas usado como referência
                pontos_brancas += posicao_b[0]

            # Se não houver nenhum ataque em todas as colunas, a pontuação é aumentada naquela posição, esta variável mostra as posições corretas das peças
            if(nao_atacadas_brancas == len(board) - l - 1):

                pontos_brancas += posicao_b[0]
        
    # Percorre todo o tabuleiro, retirando as peças brancas e certifica-se que nenhuma negra é atacada (ter em consideração que trabalhamos com números negativos)
    for l in range((len(board) - 1) - (estado_n + estado_b)):
        nao_atacadas_negras = 0
        for c in range(l + 1, len(board) - (estado_n + estado_b)):

            # Verifica se há algum ataque ou na diagonal ou na horizontal           

            # Processo iterativo para cada coluna e linha
            if (c != l) and (c != l + (c - l)) and (c != l - (c - l)):

                # Se não houver nenhum ataque na coluna que está a ser iterada a pontuação é aumentada naquela posição, esta variável é apenas usado como referência
                pontos_negras += posicao_n[0]

            # Se não houver nenhum ataque em todas as colunas, a pontuação é aumentada naquela posição, esta variável mostra as posições corretas das peças
            if(nao_atacadas_negras == len(board) - l - 1):

                pontos_negras += posicao_n[0]


    # Devolve a pontuação final como a diferença (tanto do número de peças como movimentos feitos pelas peças não atacadas)
    # Entre as brancas e as pretas multiplicando por a variável 'play' para determinar se a jogada é boa ou má 
    return (pontos_brancas - pontos_negras) * pow(-1, play)


 
def find_node(tr, id):
    if len(tr) == 0:
        return None
    if tr[0] == id:
        return tr
    for t in tr[-1]:
        aux = find_node(t, id)
        if aux is not None:
            return aux
    return None


def get_positions_directions(state, piece, p2, directions):
    ret = []
    for d in directions:
        for r in range(1, d[1]+1):
            if d[0] == 'N':
                if p2[0] - r < 0:
                    break
                if state[pos2_to_pos1([p2[0] - r, p2[1]])] == 'z':
                    ret.append([p2[0] - r, p2[1]])
                    continue

                if abs(ord(state[pos2_to_pos1([p2[0] - r, p2[1]])]) - ord(piece)) > 16:
                    ret.append([p2[0] - r, p2[1]])
                break

            if d[0] == 'S':
                if p2[0] + r > 7:
                    break
                if state[pos2_to_pos1([p2[0] + r, p2[1]])] == 'z':
                    ret.append([p2[0] + r, p2[1]])
                    continue

                if abs(ord(state[pos2_to_pos1([p2[0] + r, p2[1]])]) - ord(piece)) > 16:
                    ret.append([p2[0] + r, p2[1]])
                break
            if d[0] == 'W':
                if p2[1] - r < 0:
                    break
                if state[pos2_to_pos1([p2[0], p2[1] - r])] == 'z':
                    ret.append([p2[0], p2[1] - r])
                    continue

                if abs(ord(state[pos2_to_pos1([p2[0], p2[1] - r])]) - ord(piece)) > 16:
                    ret.append([p2[0], p2[1] - r])
                break
            if d[0] == 'E':
                if p2[1] + r > 7:
                    break
                if state[pos2_to_pos1([p2[0], p2[1] + r])] == 'z':
                    ret.append([p2[0], p2[1] + r])
                    continue

                if abs(ord(state[pos2_to_pos1([p2[0], p2[1] + r])]) - ord(piece)) > 16:
                    ret.append([p2[0], p2[1] + r])
                break
            if d[0] == 'NE':
                if p2[0] - r < 0 or p2[1] + r > 7:
                    break
                if state[pos2_to_pos1([p2[0] - r, p2[1] + r])] == 'z':
                    ret.append([p2[0] - r, p2[1] + r])
                    continue

                if abs(ord(state[pos2_to_pos1([p2[0] - r, p2[1] + r])]) - ord(piece)) > 16:
                    ret.append([p2[0] - r, p2[1] + r])
                break
            if d[0] == 'SW':
                if p2[0] + r > 7 or p2[1] - r < 0:
                    break
                if state[pos2_to_pos1([p2[0] + r, p2[1] - r])] == 'z':
                    ret.append([p2[0] + r, p2[1] - r])
                    continue

                if abs(ord(state[pos2_to_pos1([p2[0] + r, p2[1] - r])]) - ord(piece)) > 16:
                    ret.append([p2[0] + r, p2[1] - r])
                break
            if d[0] == 'NW':
                if p2[0] - r < 0 or p2[1] - r < 0:
                    break
                if state[pos2_to_pos1([p2[0] - r, p2[1] - r])] == 'z':
                    ret.append([p2[0] - r, p2[1] - r])
                    continue

                if abs(ord(state[pos2_to_pos1([p2[0] - r, p2[1] - r])]) - ord(piece)) > 16:
                    ret.append([p2[0] - r, p2[1] - r])
                break
            if d[0] == 'SE':
                if p2[0] + r > 7 or p2[1] + r > 7:
                    break
                if state[pos2_to_pos1([p2[0] + r, p2[1] + r])] == 'z':
                    ret.append([p2[0] + r, p2[1] + r])
                    continue

                if abs(ord(state[pos2_to_pos1([p2[0] + r, p2[1] + r])]) - ord(piece)) > 16:
                    ret.append([p2[0] + r, p2[1] + r])
                break
            if d[0] == 'PS':
                if p2[0] + r > 7:
                    break
                if state[pos2_to_pos1([p2[0] + r, p2[1]])] == 'z':
                    ret.append([p2[0] + r, p2[1]])
                    continue
                break
            if d[0] == 'PN':
                if p2[0] - r < 0:
                    break
                if state[pos2_to_pos1([p2[0] - r, p2[1]])] == 'z':
                    ret.append([p2[0] - r, p2[1]])
                    continue
                break
            if d[0] == 'PS2':
                if p2[0] + r <= 7 and p2[1] + 1 <= 7:
                    if state[pos2_to_pos1([p2[0] + r, p2[1] + 1])] != 'z':
                        if abs(ord(state[pos2_to_pos1([p2[0] + r, p2[1] + 1])]) - ord(piece)) > 16:
                            ret.append([p2[0] + r, p2[1] + 1])

                if p2[0] + r <= 7 and p2[1] - 1 >= 0:
                    if state[pos2_to_pos1([p2[0] + r, p2[1] - 1])] != 'z':
                        if abs(ord(state[pos2_to_pos1([p2[0] + r, p2[1] - 1])]) - ord(piece)) > 16:
                            ret.append([p2[0] + r, p2[1] - 1])
                continue
            if d[0] == 'PN2':
                if p2[0] - r >= 0 and p2[1] + 1 <= 7:
                    if state[pos2_to_pos1([p2[0] - r, p2[1] + 1])] != 'z':
                        if abs(ord(state[pos2_to_pos1([p2[0] - r, p2[1] + 1])]) - ord(piece)) > 16:
                            ret.append([p2[0] - r, p2[1] + 1])

                if p2[0] - r >= 0 and p2[1] - 1 >= 0:
                    if state[pos2_to_pos1([p2[0] - r, p2[1] - 1])] != 'z':
                        if abs(ord(state[pos2_to_pos1([p2[0] - r, p2[1] - 1])]) - ord(piece)) > 16:
                            ret.append([p2[0] - r, p2[1] - 1])
                continue
            if d[0] == 'H':
                if p2[0] - 2 >= 0 and p2[1] - 1 >= 0:
                    if state[pos2_to_pos1([p2[0] - 2, p2[1] - 1])] == 'z' or abs(ord(state[pos2_to_pos1([p2[0] - 2, p2[1] - 1])]) - ord(piece)) > 16:
                        ret.append([p2[0] - 2, p2[1] - 1])

                if p2[0] - 2 >= 0 and p2[1] + 1 <= 7:
                    if state[pos2_to_pos1([p2[0] - 2, p2[1] + 1])] == 'z' or abs(ord(state[pos2_to_pos1([p2[0] - 2, p2[1] + 1])]) - ord(piece)) > 16:
                        ret.append([p2[0] - 2, p2[1] + 1])

                if p2[0] - 1 >= 0 and p2[1] + 2 <= 7:
                    if state[pos2_to_pos1([p2[0] - 1, p2[1] + 2])] == 'z' or abs(ord(state[pos2_to_pos1([p2[0] - 1, p2[1] + 2])]) - ord(piece)) > 16:
                        ret.append([p2[0] - 1, p2[1] + 2])

                if p2[0] + 1 <= 7 and p2[1] + 2 <= 7:
                    if state[pos2_to_pos1([p2[0] + 1, p2[1] + 2])] == 'z' or abs(ord(state[pos2_to_pos1([p2[0] + 1, p2[1] + 2])]) - ord(piece)) > 16:
                        ret.append([p2[0] + 1, p2[1] + 2])

                if p2[0] + 2 <= 7 and p2[1] + 1 <= 7:
                    if state[pos2_to_pos1([p2[0] + 2, p2[1] + 1])] == 'z' or abs(ord(state[pos2_to_pos1([p2[0] + 2, p2[1] + 1])]) - ord(piece)) > 16:
                        ret.append([p2[0] + 2, p2[1] + 1])

                if p2[0] + 2 <= 7 and p2[1] - 1 >= 0:
                    if state[pos2_to_pos1([p2[0] + 2, p2[1] - 1])] == 'z' or abs(ord(state[pos2_to_pos1([p2[0] + 2, p2[1] - 1])]) - ord(piece)) > 16:
                        ret.append([p2[0] + 2, p2[1] - 1])

                if p2[0] + 1 <= 7 and p2[1] - 2 >= 0:
                    if state[pos2_to_pos1([p2[0] + 1, p2[1] - 2])] == 'z' or abs(ord(state[pos2_to_pos1([p2[0] + 1, p2[1] - 2])]) - ord(piece)) > 16:
                        ret.append([p2[0] + 1, p2[1] - 2])

                if p2[0] - 1 >= 0 and p2[1] - 2 >= 0:
                    if state[pos2_to_pos1([p2[0] - 1, p2[1] - 2])] == 'z' or abs(ord(state[pos2_to_pos1([p2[0] - 1, p2[1] - 2])]) - ord(piece)) > 16:
                        ret.append([p2[0] - 1, p2[1] - 2])
    return ret


def count_nodes(tr):
    ret = 0
    if len(tr) > 0:
        for t in tr[-1]:
            ret += count_nodes(t)
        return(1 + ret)
    return ret


def get_available_positions(state, p2, piece):
    ret = []
    if piece in ('a', 'h', 'A', 'H'):   #Tower
        aux = get_positions_directions(state, piece, p2, [['N', 7], ['S', 7], ['W', 7], ['E', 7]])
        if len(aux) > 0:
            ret.extend(aux)
        return ret

    if piece in ('c', 'f', 'C', 'F'):   #Bishop
        aux = get_positions_directions(state, piece, p2, [['NE', 7], ['SE', 7], ['NW', 7], ['SW', 7]])
        if len(aux) > 0:
            ret.extend(aux)
        return ret

    if piece in ('d', 'D'):   #Queen
        aux = get_positions_directions(state, piece, p2, [['N', 7], ['S', 7], ['W', 7], ['E', 7], ['NE', 7], ['SE', 7], ['NW', 7], ['SW', 7]])
        if len(aux) > 0:
            ret.extend(aux)
        return ret

    if piece in ('e', 'E'):   #King
        aux = get_positions_directions(state, piece, p2, [['N', 1], ['S', 1], ['W', 1], ['E', 1], ['NE', 1], ['SE', 1], ['NW', 1], ['SW', 1]])
        if len(aux) > 0:
            ret.extend(aux)
        return ret

    if piece in ('b', 'g', 'B', 'G'):  # Horse
        aux = get_positions_directions(state, piece, p2, [['H', 1]])
        if len(aux) > 0:
            ret.extend(aux)
        return ret

    # Pawn
    if ord('i') <= ord(piece) <= ord('p'):
        if p2[0] == 1:
            aux = get_positions_directions(state, piece, p2, [['PS', 2]])
            if len(aux) > 0:
                ret.extend(aux)
        else:
            aux = get_positions_directions(state, piece, p2, [['PS', 1]])
            if len(aux) > 0:
                ret.extend(aux)
        aux = get_positions_directions(state, piece, p2, [['PS2', 1]])
        if len(aux) > 0:
            ret.extend(aux)
        return ret

    if ord('I') <= ord(piece) <= ord('P'):
        if p2[0] == 6:
            aux = get_positions_directions(state, piece, p2, [['PN', 2]])
            if len(aux) > 0:
                ret.extend(aux)
        else:
            aux = get_positions_directions(state, piece, p2, [['PN', 1]])
            if len(aux) > 0:
                ret.extend(aux)
        aux = get_positions_directions(state, piece, p2, [['PN2', 1]])
        if len(aux) > 0:
            ret.extend(aux)
    return ret


def sucessor_states(state, player):
    ret = []
    
    #print('Player=%d' % player)

    for x in range(ord('a')-player*32, ord('p')-player*32+1):

        p = state.find(chr(x))
        if p < 0:
            continue
        p2 = pos1_to_pos2(p)

        pos_available = get_available_positions(state, p2, chr(x))
        # print('%c - Tot %d' % (chr(x), len(pos_available)))

        for a in pos_available:
            state_aux = list('%s' % state)
            state_aux[p] = 'z'
            if ord('i') <= x <= ord('p') and a[0] == 7:
                state_aux[pos2_to_pos1(a)] = 'd'
            elif ord('I') <= x <= ord('P') and a[0] == 0:
                state_aux[pos2_to_pos1(a)] = 'D'
            else:
                state_aux[pos2_to_pos1(a)] = chr(x)
            ret.append(''.join(state_aux))

    return ret


def insert_state_tree(tr, nv, parent):
    nd = find_node(tr, parent[0])
    if nd is None:
        return None
    nd[-1].append(nv)
    return tr



# #####################################################################################################################
# PRINT Board

pieces = ''.join(chr(9812 + x) for x in range(12))
pieces = u' ' + pieces[:6][::-1] + pieces[6:]
allbox = ''.join(chr(9472 + x) for x in range(200))
box = [allbox[i] for i in (2, 0, 12, 16, 20, 24, 44, 52, 28, 36, 60)]
(vbar, hbar, ul, ur, ll, lr, nt, st, wt, et, plus) = box

h3 = hbar * 3

# useful constant unicode strings to draw the square borders

topline = ul + (h3 + nt) * 7 + h3 + ur
midline = wt + (h3 + plus) * 7 + h3 + et
botline = ll + (h3 + st) * 7 + h3 + lr

tpl = u' {0} ' + vbar


def inter(*args):
    """Return a unicode string with a line of the chessboard.

    args are 8 integers with the values
        0 : empty square
        1, 2, 3, 4, 5, 6: white pawn, knight, bishop, rook, queen, king
        -1, -2, -3, -4, -5, -6: same black pieces
    """
    assert len(args) == 8
    return vbar + u''.join((tpl.format(pieces[a]) for a in args))


print
pieces
print
' '.join(box)
print

start_position = (
        [
            (-4, -2, -3, -5, -6, -3, -2, -4),
            (-1,) * 8,
        ] +
        [(0,) * 8] * 4 +
        [
            (1,) * 8,
            (4, 2, 3, 5, 6, 3, 2, 4),
        ]
)


def _game(position):
    yield topline
    yield inter(*position[0])
    for row in position[1:]:
        yield midline
        yield inter(*row)
    yield botline


game = lambda squares: "\n".join(_game(squares))
game.__doc__ = "Return the chessboard as a string for a given position."

# #####################################################################################################################


def get_description_piece(piece):
    if ord(piece) < 97:
        ret = 'Black '
    else:
        ret = 'White '
    if piece.lower() in ('a', 'h'):
        ret = ret + 'Tower'
    elif piece.lower() in ('b', 'g'):
        ret = ret + 'Horse'
    elif piece.lower() in ('c', 'f'):
        ret = ret + 'Bishop'
    elif piece.lower() == 'd':
        ret = ret + 'Queen'
    elif piece.lower() == 'e':
        ret = ret + 'King'
    else:
        ret = ret + 'Pawn'
    return ret


def description_move(prev, cur, idx, nick):
    #print('description_move()')
    ret = 'Move [%d - %s]: ' % (idx, nick)

    cur_blank = [i for i, ltr in enumerate(cur) if ltr == 'z']
    prev_not_blank = [i for i, ltr in enumerate(prev) if ltr != 'z']
    #print(cur_blank)
    #print(prev_not_blank)
    moved = list(set(cur_blank) & set(prev_not_blank))
    #print(moved)
    moved = moved[0]

    desc_piece = get_description_piece(prev[moved])

    fr = pos1_to_pos2(moved)
    to = pos1_to_pos2(cur.find(prev[moved]))
    #print(fr)
    #print(to)

    ret = ret + desc_piece + ' (%d, %d) --> (%d, %d)' % (fr[0], fr[1], to[0], to[1])
    if prev[pos2_to_pos1(to)] != 'z':
        desc_piece = get_description_piece(prev[pos2_to_pos1(to)])
        ret = ret + ' eaten ' + desc_piece
    return ret


def show_board(prev, cur, idx, nick):
    print('print_board(obj: %f)...' % idx)
    state_show = []
    for r in range(0, 8):
        row = []
        for c in range(0, 8):
            if cur[pos2_to_pos1([r, c])] == 'z':
                row.append(0)

            if cur[pos2_to_pos1([r, c])] == 'a':
                row.append(-4)
            if cur[pos2_to_pos1([r, c])] == 'b':
                row.append(-2)
            if cur[pos2_to_pos1([r, c])] == 'c':
                row.append(-3)
            if cur[pos2_to_pos1([r, c])] == 'd':
                row.append(-5)
            if cur[pos2_to_pos1([r, c])] == 'e':
                row.append(-6)
            if cur[pos2_to_pos1([r, c])] == 'f':
                row.append(-3)
            if cur[pos2_to_pos1([r, c])] == 'g':
                row.append(-2)
            if cur[pos2_to_pos1([r, c])] == 'h':
                row.append(-4)
            if ord('i') <= ord(cur[pos2_to_pos1([r, c])]) <= ord('p'):
                row.append(-1)

            if cur[pos2_to_pos1([r, c])] == 'A':
                row.append(4)
            if cur[pos2_to_pos1([r, c])] == 'B':
                row.append(2)
            if cur[pos2_to_pos1([r, c])] == 'C':
                row.append(3)
            if cur[pos2_to_pos1([r, c])] == 'D':
                row.append(5)
            if cur[pos2_to_pos1([r, c])] == 'E':
                row.append(6)
            if cur[pos2_to_pos1([r, c])] == 'F':
                row.append(3)
            if cur[pos2_to_pos1([r, c])] == 'G':
                row.append(2)
            if cur[pos2_to_pos1([r, c])] == 'H':
                row.append(4)
            if ord('I') <= ord(cur[pos2_to_pos1([r, c])]) <= ord('P'):
                row.append(1)
        state_show.append(tuple(row))

    ret = game(state_show) + '\n'

    if prev is None:
        return ret
    ret = ret + description_move(prev, cur, idx, nick)

    return ret


def expand_tree(tr, dep, n, play):
    if n == 0:
        return tr
    suc = sucessor_states(tr[0], play)
    for s in suc:
        tr = insert_state_tree(tr, expand_tree([s,  random.random(), dep+1, 0, f_obj(s, play), []], dep+1, n-1, 1 - play), tr)
    return tr


def show_tree(tr, play, nick, depth):
    if len(tr) == 0:
        return
    print('DEPTH %d' % depth)
    print('%s' % show_board(None, tr[0], f_obj(tr[0], play), nick))
    for t in tr[-1]:
        show_tree(t, play, nick, depth+1)



def get_father(tr, st):
    if len(tr) == 0:
        return None
    for sun in tr[-1]:
        if sun[1] == st[1]:
            return tr

    for sun in tr[-1]:
        aux = get_father(sun, st)
        if aux is not None:
            return aux

    return None


def get_next_move(tree, st):
    old = None
    while get_father(tree, st) is not None:
        old = st
        st = get_father(tree, st)
    return old



def minimax_alpha_beta(tr, d, play, max_player, alpha, beta):
    if d == 0 or len(tr[-1]) == 0:
        return tr, f_obj(tr[0], play)

    ret = math.inf * pow(-1, max_player)
    ret_nd = tr
    for s in tr[-1]:
        aux, val = minimax_alpha_beta(s, d - 1, play, not max_player, alpha, beta)
        if max_player:
            if val > ret:
                ret = val
                ret_nd = aux
            alpha = max(alpha, ret)
        else:
            if val < ret:
                ret = val
                ret_nd = aux
            beta = min(beta, ret)
        if beta <= alpha:
            break

    return ret_nd, ret
    


def decide_move(board, play, nick):
    
    states = expand_tree([board, random.random(), 0, f_obj(board, play), []], 0, depth_analysis, play)    # [board, hash, depth, g(), f_obj(), [SUNS]]
    
    # show_tree(states, play, nick, 0)
    print('Total nodes in the tree: %d' % count_nodes(states))
            
    choice, value = minimax_alpha_beta(states, depth_analysis, play, True, -math.inf, math.inf)
        
    # print('Choose f()=%f' % value)
    # print('State_%s_' % choice[0])
    
    
    next_move = get_next_move(states, choice)
    
    # print('Next_%s_' % next_move[0])
    # input('Trash')

    return next_move[0]





client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)      #socket initialization
client.connect((sys.argv[1], int(sys.argv[2])))                             #connecting client to server

hello_msg = '%s_%s' % (sys.argv[4], sys.argv[3])
client.send(hello_msg.encode('ascii'))

nickname = sys.argv[3]

player = int(sys.argv[4])

while True:                                                 #making valid connection
    while True:
        message = client.recv(1024).decode('ascii')
        if len(message) > 0:
            break

    if interactive_flag:
        row_from = int(input('Row from > '))
        col_from = int(input('Col from > '))
        row_to = int(input('Row to > '))
        col_to = int(input('Col to > '))

        p_from = pos2_to_pos1([row_from, col_from])
        p_to = pos2_to_pos1([row_to, col_to])

        if (0 <= p_from <= 63) and (0 <= p_to <= 63):
            message = list(message)
            aux = message[p_from]
            message[p_from] = 'z'
            message[p_to] = aux
            message = ''.join(message)
    else:
        message = decide_move(message, player, nickname)

    client.send(message.encode('ascii'))
