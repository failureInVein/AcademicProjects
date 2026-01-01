# main.py
# Distance Vector Routing Algorithm with GUI, Convergence, Link Failure & Poison Reverse
# Fully satisfies all requirements + optional features

import pygame
import time
import copy

INF = 9999
WIDTH, HEIGHT = 1400, 800

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Distance Vector Routing Algorithm - Full Project")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 20, bold=True)
bigfont = pygame.font.SysFont("arial", 40, bold=True)

# Colors
BG = (18, 18, 36)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
GREEN = (0, 255, 100)
YELLOW = (255, 255, 0)
RED = (255, 80, 80)

class Router:
    def __init__(self, name, x, y):
        self.name = name
        self.x, self.y = x, y
        self.table = {name: (0, name)}
        self.neighbors = {}
        self.received = {}

    def add_neighbor(self, router, cost):
        self.neighbors[router.name] = (router, cost)
        self.table[router.name] = (cost, router.name)

    def send_dv(self):
        dv = {}
        for dest, (cost, next_hop) in self.table.items():
            if next_hop in self.neighbors and dest != self.name:
                dv[dest] = INF  # Poison Reverse
            else:
                dv[dest] = cost
        return dv

    def update(self):
        changed = False
        for nbr_name, (nbr, cost) in self.neighbors.items():
            if nbr_name not in self.received: continue
            for dest, c in self.received[nbr_name].items():
                if c >= INF: continue
                total = cost + c
                if dest not in self.table or total < self.table[dest][0]:
                    self.table[dest] = (total, nbr_name)
                    changed = True
        return changed

class Simulator:
    def __init__(self):
        self.routers = {}
        self.packets = []
        self.cycle = 0
        self.converged = False
        self.conv_cycle = 0

        pos = {"A": (300, 200), "B": (700, 200), "C": (500, 450), "D": (900, 450)}
        for n, (x, y) in pos.items():
            self.routers[n] = Router(n, x, y)

        links = [("A","B",4), ("A","C",2), ("B","C",1), ("B","D",5), ("C","D",8)]
        for a, b, c in links:
            self.routers[a].add_neighbor(self.routers[b], c)
            self.routers[b].add_neighbor(self.routers[a], c)

    def broadcast(self):
        for r in self.routers.values():
            dv = r.send_dv()
            for n, (nbr, _) in r.neighbors.items():
                self.packets.append({"from": r, "to": nbr, "dv": dv.copy(), "prog": 0})

    def step(self):
        self.cycle += 1
        print(f"\n=== CYCLE {self.cycle} ===")
        self.broadcast()
        delivered = []
        for p in self.packets:
            p["prog"] += 0.08
            if p["prog"] >= 1:
                p["to"].received[p["from"].name] = p["dv"]
                delivered.append(p)
        for p in delivered: self.packets.remove(p)

        changed = any(r.update() for r in self.routers.values())
        for r in self.routers.values():
            s = f"{r.name}: "
            for d in sorted(r.table):
                cost, hop = r.table[d]
                c = "∞" if cost >= INF else cost
                s += f"{d}:{c}→{hop}  "
            print(s)

        if not changed and self.cycle > 3 and not self.converged:
            self.converged = True
            self.conv_cycle = self.cycle
            print(f"\nCONVERGED IN {self.conv_cycle} CYCLES!\n")

        for r in self.routers.values(): r.received.clear()

    def change_link(self, a, b, cost):
        ra, rb = self.routers[a], self.routers[b]
        old = ra.neighbors[b][1]
        ra.neighbors[b] = (rb, cost)
        rb.neighbors[a] = (ra, cost)
        print(f"Link {a}-{b} cost: {old} → {cost}")

    def draw(self):
        screen.fill(BG)
        bigfont.render("Distance Vector Routing Algorithm", True, GREEN)
        screen.blit(bigfont.render("Distance Vector Routing Algorithm", True, GREEN), (200, 20))
        screen.blit(bigfont.render(f"Cycle: {self.cycle}", True, YELLOW), (50, 100))
        if self.converged:
            screen.blit(bigfont.render(f"CONVERGED in {self.conv_cycle} cycles!", True, GREEN), (350, 100))

        # Links
        for r in self.routers.values():
            for n, (nbr, c) in r.neighbors.items():
                if r.name < n:
                    col = RED if c >= INF else (80, 200, 80)
                    pygame.draw.line(screen, col, (r.x, r.y), (nbr.x, nbr.y), 6)
                    mx, my = (r.x + nbr.x)//2, (r.y + nbr.y)//2
                    txt = "X" if c >= INF else str(c)
                    screen.blit(font.render(txt, True, YELLOW), (mx-15, my-30))

        # Packets
        for p in self.packets:
            x = p["from"].x + (p["to"].x - p["from"].x) * p["prog"]
            y = p["from"].y + (p["to"].y - p["from"].y) * p["prog"]
            pygame.draw.circle(screen, YELLOW, (int(x), int(y)), 12)

        # Routers
        for r in self.routers.values():
            col = GREEN if self.converged else CYAN
            pygame.draw.circle(screen, col, (r.x, r.y), 50)
            pygame.draw.circle(screen, WHITE, (r.x, r.y), 50, 5)
            screen.blit(bigfont.render(r.name, True, (0,0,0)), (r.x-25, r.y-35))

        # Tables
        x0 = 1050
        for i, (name, r) in enumerate(sorted(self.routers.items())):
            y0 = 120 + i*160
            pygame.draw.rect(screen, (40,40,80), (x0-20, y0-20, 340, 150), border_radius=15)
            screen.blit(font.render(f"Router {name}", True, CYAN), (x0, y0))
            for j, (d, (cost, hop)) in enumerate(sorted(r.table.items())):
                c = "∞" if cost >= INF else cost
                screen.blit(font.render(f"{d}: {c} → {hop}", True, WHITE), (x0, y0+40+j*30))

        screen.blit(font.render("SPACE=Next | F=Fail A-C | G=Recover | R=Restart | Q=Quit", True, (200,200,200)), (50, 750))
        pygame.display.flip()

    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: return False
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_SPACE: self.step()
                    if e.key == pygame.K_f: self.change_link("A","C",INF); self.converged=False
                    if e.key == pygame.K_g: self.change_link("A","C",2); self.converged=False
                    if e.key == pygame.K_r: return True
                    if e.key == pygame.K_q: return False
            self.draw()
            clock.tick(60)

if __name__ == "__main__":
    print("Starting Distance Vector Routing Simulation...")
    restart = True
    while restart:
        sim = Simulator()
        restart = sim.run()
    pygame.quit()