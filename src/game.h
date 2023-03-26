// This file defines the game logic.
#ifndef GAME_H
#define GAME_H

#include <cassert>
#include <string>

inline char get_left(char c) { return c < 18 ? 3 : c < 36 ? 4 : 8; }
inline char get_right(char c) { return c % 6 < 2 ? 2 : c % 6 < 4 ? 6 : 7; }
inline char get_mid(char c) { return c % 18 < 6 ? 1 : c % 18 < 12 ? 5 : 9; }

struct Game {
  char square = -1, turn = 0;
  short score = 0, board[20];
  unsigned long long flag = -1;

  Game() { std::fill(board, board + 20, -1); }

  bool is_ended() const { return turn >= 20; }
  int get_score() const { return score; }
  bool is_chance() const { return -1 == square; }

  void get_chances(std::string& a, float p[28]) const {
    double cnt = 0;
    a.reserve(28);
    for (int i = 0; i < 54; i += 2) {
      if (flag & (1ull << i)) {
        a += char(i);
        p[a.size() - 1] = 2;
        cnt += 2;
      } else if (flag & (1ull << i + 1)) {
        a += char(i + 1);
        p[a.size() - 1] = 1;
        cnt++;
      }
    }
    if (flag & (1ull << 54)) {
      int n = turn;
      float p0 = (90 - 2 * n) / (n * n - 91 * n + 1540.);
      for (int i = a.size() - 1; i >= 0; i--) {
        p[i] = p[i] / cnt * (1 - p0);
      }
      a += char(54);
      p[a.size() - 1] = p0;
    } else if (flag & (1ull << 55)) {
      int n = turn;
      float p0 = 1. / (46 - n);
      for (int i = a.size() - 1; i >= 0; i--) {
        p[i] = p[i] / cnt * (1 - p0);
      }
      a += char(55);
      p[a.size() - 1] = p0;
    } else {
      for (int i = a.size() - 1; i >= 0; i--) {
        p[i] /= cnt;
      }
    }
    // for (int i = 0; i < a.size(); i++) {
    //   printf("p[%d] = %f\n", i, p[i]);
    // }
  }

  void get_actions(std::string& a) const {
    for (int i = 0; i < 20; i++) {
      if (board[i] == -1) a += char(i);
    }
  }

  bool is_legal(char a) const { return board[a] == -1; }

  void step(char a) {
    if (square == -1) {
      square = a;
      assert(flag & (1ull << a));
      flag ^= 1ull << a;
      return;
    }
    board[a] = square;
    square = -1;
    if (++turn == 20) _get_final_score();
  }

  void redo(char a) {
    if (square == -1) {
      square = board[a];
      turn--;
      board[a] = -1;
    } else {
      flag |= 1ull << square;
      square = -1;
    }
  }

  void _get_final_score();
};
#endif