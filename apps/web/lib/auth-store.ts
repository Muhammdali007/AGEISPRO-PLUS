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

const AUTH_HINT_KEY = "aegispro.authenticated";
const ACCESS_TOKEN_KEY = "aegispro.access_token";
const REFRESH_TOKEN_KEY = "aegispro.refresh_token";

export const useAuthStore = create<AuthState>((set) => ({
  accessToken: null,
  refreshToken: null,
  user: null,
  hydrated: false,
  setTokens: (tokens) => {
    localStorage.setItem(AUTH_HINT_KEY, "true");
    localStorage.setItem(ACCESS_TOKEN_KEY, tokens.access_token);
    localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refresh_token);
    set({
      accessToken: tokens.access_token,
      refreshToken: tokens.refresh_token,
      hydrated: true
    });
  },
  setUser: (user) => set({ user }),
  hydrate: () => {
    const authenticated = localStorage.getItem(AUTH_HINT_KEY) === "true";
    const accessToken = localStorage.getItem(ACCESS_TOKEN_KEY);
    const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);
    set({
      accessToken: authenticated ? accessToken : null,
      refreshToken: authenticated ? refreshToken : null,
      hydrated: true
    });
  },
  logout: () => {
    localStorage.removeItem(AUTH_HINT_KEY);
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    set({ accessToken: null, refreshToken: null, user: null, hydrated: true });
  }
}));
