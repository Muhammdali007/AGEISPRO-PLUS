"use client";

import { create } from "zustand";
import type { CurrentUser, TokenPair } from "@/lib/api";

type AuthState = {
  accessToken: string | null;
  refreshToken: string | null;
  user: CurrentUser | null;
  hydrated: boolean;
  setTokens: (tokens: TokenPair) => void;
  setUser: (user: CurrentUser) => void;
  hydrate: () => void;
  logout: () => void;
};

const ACCESS_KEY = "aegispro.access";
const REFRESH_KEY = "aegispro.refresh";

export const useAuthStore = create<AuthState>((set) => ({
  accessToken: null,
  refreshToken: null,
  user: null,
  hydrated: false,
  setTokens: (tokens) => {
    localStorage.setItem(ACCESS_KEY, tokens.access_token);
    localStorage.setItem(REFRESH_KEY, tokens.refresh_token);
    set({
      accessToken: tokens.access_token,
      refreshToken: tokens.refresh_token,
      hydrated: true
    });
  },
  setUser: (user) => set({ user }),
  hydrate: () => {
    const accessToken = localStorage.getItem(ACCESS_KEY);
    const refreshToken = localStorage.getItem(REFRESH_KEY);
    set({
      accessToken,
      refreshToken,
      hydrated: true
    });
  },
  logout: () => {
    localStorage.removeItem(ACCESS_KEY);
    localStorage.removeItem(REFRESH_KEY);
    set({ accessToken: null, refreshToken: null, user: null, hydrated: true });
  }
}));
