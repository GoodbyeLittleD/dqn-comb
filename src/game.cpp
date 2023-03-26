#include "game.h"

void Game::_get_final_score(){
    score = board[0] < 54 ? get_left(board[0]) + get_mid(board[0]) + get_right(board[0]) : 30;

    int x = board[1] < 54 ? get_mid(board[1]) : -1;
    int y = board[2] < 54 ? get_mid(board[2]) : x;
    int z = board[3] < 54 ? get_mid(board[3]) : y;
    if ((x == -1 || x == y) && (y == -1 || y == z)) {
        score += 3 * z;
    }
    x = board[4] < 54 ? get_mid(board[4]) : -1;
    y = board[5] < 54 ? get_mid(board[5]) : x;
    z = board[6] < 54 ? get_mid(board[6]) : y;
    int u = board[7] < 54 ? get_mid(board[7]) : z;
    if ((x == -1 || x == y) && (y == -1 || y == z) && (z == u)) {
        score += 4 * z;
    }
    x = board[8] < 54 ? get_mid(board[8]) : -1;
    y = board[9] < 54 ? get_mid(board[9]) : x;
    z = board[10] < 54 ? get_mid(board[10]) : y;
    u = board[11] < 54 ? get_mid(board[11]) : z;
    int v = board[12] < 54 ? get_mid(board[12]) : u;
    if ((x == -1 || x == y) && (y == -1 || y == z) && (z == u) && (u == v)) {
        score += 5 * z;
    }
    x = board[13] < 54 ? get_mid(board[13]) : -1;
    y = board[14] < 54 ? get_mid(board[14]) : x;
    z = board[15] < 54 ? get_mid(board[15]) : y;
    u = board[16] < 54 ? get_mid(board[16]) : z;
    if ((x == -1 || x == y) && (y == -1 || y == z) && (z == u)) {
        score += 4 * z;
    }
    x = board[17] < 54 ? get_mid(board[17]) : -1;
    y = board[18] < 54 ? get_mid(board[18]) : x;
    z = board[19] < 54 ? get_mid(board[19]) : y;
    if ((x == -1 || x == y) && (y == -1 || y == z)) {
        score += 3 * z;
    }

    x = board[3] < 54 ? get_left(board[3]) : -1;
    y = board[7] < 54 ? get_left(board[7]) : x;
    z = board[12] < 54 ? get_left(board[12]) : y;
    if ((x == -1 || x == y) && (y == -1 || y == z)) {
        score += 3 * z;
    }
    x = board[2] < 54 ? get_left(board[2]) : -1;
    y = board[11] < 54 ? get_left(board[11]) : x;
    z = board[6] < 54 ? get_left(board[6]) : y;
    u = board[16] < 54 ? get_left(board[16]) : z;
    if ((x == -1 || x == y) && (y == -1 || y == z) && (z == u)) {
        score += 4 * z;
    }
    x = board[1] < 54 ? get_left(board[1]) : -1;
    y = board[5] < 54 ? get_left(board[5]) : x;
    z = board[10] < 54 ? get_left(board[10]) : y;
    u = board[15] < 54 ? get_left(board[15]) : z;
    v = board[19] < 54 ? get_left(board[19]) : u;
    if ((x == -1 || x == y) && (y == -1 || y == z) && (z == u) && (u == v)) {
        score += 5 * z;
    }
    x = board[4] < 54 ? get_left(board[4]) : -1;
    y = board[14] < 54 ? get_left(board[14]) : x;
    z = board[9] < 54 ? get_left(board[9]) : y;
    u = board[18] < 54 ? get_left(board[18]) : z;
    if ((x == -1 || x == y) && (y == -1 || y == z) && (z == u)) {
        score += 4 * z;
    }
    x = board[8] < 54 ? get_left(board[8]) : -1;
    y = board[13] < 54 ? get_left(board[13]) : x;
    z = board[17] < 54 ? get_left(board[17]) : y;
    if ((x == -1 || x == y) && (y == -1 || y == z)) {
        score += 3 * z;
    }

    x = board[1] < 54 ? get_right(board[1]) : -1;
    y = board[4] < 54 ? get_right(board[4]) : x;
    z = board[8] < 54 ? get_right(board[8]) : y;
    if ((x == -1 || x == y) && (y == -1 || y == z)) {
        score += 3 * z;
    }
    x = board[2] < 54 ? get_right(board[2]) : -1;
    y = board[5] < 54 ? get_right(board[5]) : x;
    z = board[9] < 54 ? get_right(board[9]) : y;
    u = board[13] < 54 ? get_right(board[13]) : z;
    if ((x == -1 || x == y) && (y == -1 || y == z) && (z == u)) {
        score += 4 * z;
    }
    x = board[3] < 54 ? get_right(board[3]) : -1;
    y = board[6] < 54 ? get_right(board[6]) : x;
    z = board[10] < 54 ? get_right(board[10]) : y;
    u = board[14] < 54 ? get_right(board[14]) : z;
    v = board[17] < 54 ? get_right(board[17]) : u;
    if ((x == -1 || x == y) && (y == -1 || y == z) && (z == u) && (u == v)) {
        score += 5 * z;
    }
    x = board[7] < 54 ? get_right(board[7]) : -1;
    y = board[11] < 54 ? get_right(board[11]) : x;
    z = board[15] < 54 ? get_right(board[15]) : y;
    u = board[18] < 54 ? get_right(board[18]) : z;
    if ((x == -1 || x == y) && (y == -1 || y == z) && (z == u)) {
        score += 4 * z;
    }
    x = board[12] < 54 ? get_right(board[12]) : -1;
    y = board[16] < 54 ? get_right(board[16]) : x;
    z = board[19] < 54 ? get_right(board[19]) : y;
    if ((x == -1 || x == y) && (y == -1 || y == z)) {
        score += 3 * z;
    }
}