#!/usr/bin/env python3
"""
distance_vector_sim.py

Level-3 Distance Vector Routing Simulator (Bellman-Ford style)

Features:
- Router class with routing table, neighbors, Bellman-Ford updates
- Message-based exchanges via NetworkSimulator (event-driven ticks)
- Split horizon and poison reverse options
- Route poisoning and configurable MAX_HOP (infinity)
- Triggered updates and periodic updates
- Convergence detection and logging
- Optional visualization (networkx + matplotlib) if installed
"""

from __future__ import annotations
import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Any, Set
import copy
import math
import random
import sys
import pprint
import statistics

# --- Configuration constants ---
INFINITY = 16  # adopt RIP-like infinity by default; configurable per-simulator
DEFAULT_PERIODIC_INTERVAL = 5  # ticks between periodic full updates
DEFAULT_MESSAGE_DELAY = 1  # ticks for message delivery (simple model)
VERBOSE = False  # toggle internal prints


@dataclass
class DVEntry:
    distance: int
    next_hop: Optional[str]
    # optional metadata for debugging/convergence
    poisoned: bool = False  # if distance is poisoned (infinite)
    last_updated_tick: int = -1


@dataclass(order=True)
class Event:
    tick: int
    priority: int
    payload: Any = field(compare=False)


@dataclass
class Message:
    src: str
    dst: str
    vector: Dict[str, int]  # destination -> reported distance
    poisoned_info: Optional[Set[str]] = None  # which destinations were poisoned by src
    triggered: bool = False


class Router:
    """
    Router implementing Distance Vector logic (Bellman-Ford variant).
    """

    def __init__(
        self,
        name: str,
        neighbors: Dict[str, int],
        sim: "NetworkSimulator",
        max_hop: int = INFINITY,
        split_horizon: bool = True,
        poison_reverse: bool = True,
    ):
        self.name = name
        self.sim = sim
        # neighbor -> cost
        self.neighbors: Dict[str, int] = dict(neighbors)
        # routing table: dest -> DVEntry
        self.table: Dict[str, DVEntry] = {}
        self.max_hop = max_hop
        self.split_horizon = split_horizon
        self.poison_reverse = poison_reverse
        # initialize table with direct neighbors and self
        self._init_table()
        # pending messages (for triggered updates batching)
        self.pending_triggered: bool = False

    def _init_table(self):
        # self to self distance 0
        self.table[self.name] = DVEntry(distance=0, next_hop=self.name, last_updated_tick=self.sim.current_tick)
        for n, cost in self.neighbors.items():
            self.table[n] = DVEntry(distance=cost, next_hop=n, last_updated_tick=self.sim.current_tick)

    def get_distance(self, dest: str) -> int:
        e = self.table.get(dest)
        return e.distance if e else self.max_hop

    def make_vector(self) -> Tuple[Dict[str, int], Set[str]]:
        """
        Return the distance vector to send to neighbors.
        Also return a set of poisoned destinations (if we mark any as poisoned).
        """
        vector = {}
        poisoned = set()
        for dest, entry in self.table.items():
            d = entry.distance
            # If we've exceeded max_hop, advertise as infinity
            if d >= self.max_hop:
                vector[dest] = self.max_hop
                poisoned.add(dest)
            else:
                vector[dest] = d
        return vector, poisoned

    def receive_message(self, msg: Message):
        """Called by simulator when delivery occurs."""
        if VERBOSE:
            print(f"[{self.sim.current_tick}] Router {self.name} received vector from {msg.src}: {msg.vector}")
        changed = False
        # For each destination in received vector, attempt to relax via src
        for dest, reported_dist in msg.vector.items():
            if dest == self.name:
                continue  # skip self
            if reported_dist >= self.max_hop:
                # neighbor says dest unreachable
                via_cost = self.max_hop
            else:
                # cost to reach dest via src = cost(self->src) + reported_dist
                link_cost = self.neighbors.get(msg.src, math.inf)
                via_cost = link_cost + reported_dist

            current = self.table.get(dest)
            if current is None:
                # new destination discovered
                new_dist = min(via_cost, self.max_hop)
                next_hop = msg.src if new_dist < self.max_hop else None
                self.table[dest] = DVEntry(distance=new_dist, next_hop=next_hop, poisoned=(new_dist >= self.max_hop), last_updated_tick=self.sim.current_tick)
                changed = True
            else:
                # Update logic:
                # If route via msg.src improves distance -> adopt
                if via_cost < current.distance:
                    old = (current.distance, current.next_hop)
                    current.distance = min(via_cost, self.max_hop)
                    current.next_hop = msg.src
                    current.poisoned = current.distance >= self.max_hop
                    current.last_updated_tick = self.sim.current_tick
                    changed = True
                else:
                    # if current route uses msg.src as next_hop, and neighbor now reports worse/unreachable, we may need to update
                    if current.next_hop == msg.src:
                        # recompute best alternative by scanning neighbors (Bellman-Ford)
                        best_dist = self.max_hop
                        best_next = None
                        for nb, nb_cost in self.neighbors.items():
                            # avoid using a missing neighbor in case link was removed
                            # We need the last known vector from nb; simulator maintains last_sent_vectors
                            nb_vector = self.sim.last_sent_vectors.get(nb, {})
                            reported_by_nb = nb_vector.get(dest, self.max_hop)
                            candidate = nb_cost + reported_by_nb
                            if candidate < best_dist:
                                best_dist = candidate
                                best_next = nb
                        # if best differs from current -> update
                        if best_next != current.next_hop or best_dist != current.distance:
                            current.distance = min(best_dist, self.max_hop)
                            current.next_hop = best_next
                            current.poisoned = current.distance >= self.max_hop
                            current.last_updated_tick = self.sim.current_tick
                            changed = True
        # If changed -> schedule triggered update
        if changed:
            self.schedule_triggered_update()

    def schedule_triggered_update(self):
        # mark that this router has a triggered update pending
        self.pending_triggered = True
        # schedule an immediate send with small jitter to avoid simultaneity collisions
        jitter = random.randint(0, 1)
        self.sim.schedule_send(self.name, delay=1 + jitter, triggered=True)

    def periodic_send(self):
        """Send full vector to all neighbors (used by periodic timers)."""
        self._send_vector(triggered=False)

    def _send_vector(self, triggered: bool = False):
        vector, poisoned = self.make_vector()
        # store last_sent for others to query if needed
        self.sim.last_sent_vectors[self.name] = vector.copy()

        # For each neighbor, potentially apply split horizon / poison reverse
        for nb in list(self.neighbors.keys()):
            advert = {}
            # apply split horizon: do not advertise routes learned via nb back to nb
            for dest, dist in vector.items():
                # If split_horizon is enabled and next_hop for dest is nb, skip/poison
                entry = self.table.get(dest)
                if self.split_horizon and entry and entry.next_hop == nb:
                    if self.poison_reverse:
                        advert[dest] = self.max_hop
                    else:
                        # skip this dest entirely
                        continue
                else:
                    advert[dest] = dist
            msg = Message(src=self.name, dst=nb, vector=advert, poisoned_info=set([d for d, dd in advert.items() if dd >= self.max_hop]), triggered=triggered)
            self.sim.send_message(msg, delay=DEFAULT_MESSAGE_DELAY)

        # reset triggered flag after sending
        self.pending_triggered = False

    # External API
    def handle_tick(self, tick: int):
        """
        Called every tick by the simulator. Router decides whether to send periodic updates.
        For triggered updates, the simulator schedules a send already.
        """
        # nothing to do each tick beyond scheduled sends; periodic sends handled by simulator timer

    def update_link_cost(self, neighbor: str, cost: Optional[int]):
        """
        Update neighbor link cost. If cost is None -> remove neighbor (link down).
        """
        if cost is None:
            # remove neighbor
            if neighbor in self.neighbors:
                del self.neighbors[neighbor]
        else:
            self.neighbors[neighbor] = cost

        # Update direct route entry
        if cost is None:
            # mark neighbor unreachable in table
            entry = self.table.get(neighbor)
            if entry:
                entry.distance = self.max_hop
                entry.next_hop = None
                entry.poisoned = True
                entry.last_updated_tick = self.sim.current_tick
        else:
            # set direct neighbor cost
            self.table[neighbor] = DVEntry(distance=cost, next_hop=neighbor, poisoned=False, last_updated_tick=self.sim.current_tick)

        # schedule triggered update
        self.schedule_triggered_update()

    def get_routing_table_snapshot(self):
        # return a deep-copy dictionary for logging/printing
        return {d: (e.distance, e.next_hop, e.poisoned) for d, e in self.table.items()}


class NetworkSimulator:
    """
    Event-driven simulator that handles message delivery and orchestrates routers.
    """

    def __init__(self, periodic_interval: int = DEFAULT_PERIODIC_INTERVAL, max_hop: int = INFINITY):
        self.routers: Dict[str, Router] = {}
        self.event_queue: List[Event] = []
        self.current_tick: int = 0
        self.periodic_interval = periodic_interval
        self.max_hop = max_hop
        # hold last vector each router sent (for router-local checks)
        self.last_sent_vectors: Dict[str, Dict[str, int]] = {}
        # for logging changes
        self.history: List[Dict[str, Dict[str, Tuple[int, Optional[str], bool]]]] = []
        # schedule of periodic updates (simple)
        self.next_periodic_tick = self.periodic_interval
        self._event_counter = 0

    # --- router lifecycle ---
    def add_router(self, name: str, neighbors: Dict[str, int], **router_opts):
        if name in self.routers:
            raise ValueError(f"Router {name} already present")
        r = Router(name=name, neighbors=neighbors, sim=self, max_hop=self.max_hop, **router_opts)
        self.routers[name] = r
        # store initial vectors
        v, _ = r.make_vector()
        self.last_sent_vectors[name] = v
        return r

    def send_message(self, msg: Message, delay: int = 1):
        """Schedule message delivery after 'delay' ticks."""
        deliver_tick = self.current_tick + delay
        self._enqueue_event(deliver_tick, 1, ("deliver", msg))

    def schedule_send(self, router_name: str, delay: int = 1, triggered: bool = False):
        """Schedule a router to send its vector at given tick."""
        deliver_tick = self.current_tick + delay
        self._enqueue_event(deliver_tick, 2, ("send", router_name, triggered))

    def _enqueue_event(self, tick: int, priority: int, payload: Any):
        self._event_counter += 1
        ev = Event(tick=tick, priority=self._event_counter, payload=payload)
        heapq.heappush(self.event_queue, ev)

    # --- simulation loop ---
    def run(self, max_ticks: int = 1000, until_converged: bool = True, debug: bool = False):
        """
        Run the simulation. If until_converged is True, stops when the network reaches convergence.
        Returns stats about convergence: (converged, tick)
        """
        converged_at: Optional[int] = None
        if debug:
            print(f"[SIM] Starting simulation up to {max_ticks} ticks")

        # initial: schedule periodic sends at tick 0 to bootstrap
        for rname in self.routers.keys():
            self.schedule_send(rname, delay=0, triggered=False)

        for tick_limit in range(max_ticks):
            if not self.event_queue:
                # nothing scheduled; supply periodic tick
                self.current_tick += 1
                self._periodic_tick_actions()
                if debug:
                    print(f"[SIM] tick {self.current_tick}: no events -> periodic")
            else:
                # process next event(s) for this tick
                ev = heapq.heappop(self.event_queue)
                self.current_tick = ev.tick
                # process all events with this tick
                batch = [ev]
                while self.event_queue and self.event_queue[0].tick == ev.tick:
                    batch.append(heapq.heappop(self.event_queue))
                # sort by payload priority as inserted
                for e in batch:
                    self._handle_event(e.payload)
                # periodics: if reached next periodic tick, schedule periodic updates
                if self.current_tick >= self.next_periodic_tick:
                    self._periodic_tick_actions()
                    # advance periodic schedule
                    self.next_periodic_tick = self.current_tick + self.periodic_interval

            # log snapshot
            snapshot = {rname: self.routers[rname].get_routing_table_snapshot() for rname in self.routers}
            self.history.append(snapshot)

            # check convergence: no router pending triggered and no events deliverable that would change tables
            if until_converged:
                if self._check_converged():
                    converged_at = self.current_tick
                    if debug:
                        print(f"[SIM] Converged at tick {self.current_tick}")
                    break

            if debug and (self.current_tick % 10 == 0):
                print(f"[SIM] tick {self.current_tick}")

        return (converged_at is not None, converged_at)

    def _handle_event(self, payload):
        typ = payload[0]
        if typ == "deliver":
            msg: Message = payload[1]
            # if destination router exists and there is still a link from dst to src
            dst_router = self.routers.get(msg.dst)
            if dst_router:
                # deliver
                dst_router.receive_message(msg)
        elif typ == "send":
            router_name = payload[1]
            triggered = bool(payload[2])
            r = self.routers.get(router_name)
            if r:
                # the router sends vector to all neighbors
                r._send_vector(triggered=triggered)
        else:
            raise RuntimeError(f"Unknown event payload type {typ}")

    def _periodic_tick_actions(self):
        # Periodic updates: all routers send full vector
        for rname, r in self.routers.items():
            self.schedule_send(rname, delay=0, triggered=False)

    def _check_converged(self) -> bool:
        # Convergence heuristic:
        # - No pending triggered updates
        # - event_queue contains only sends that are periodic (but since periodic sends are scheduled we consider them)
        # Simpler: check last two snapshots identical
        if len(self.history) < 2:
            return False
        return self.history[-1] == self.history[-2]

    # helper: change a link cost between two routers (a,b). if new_cost is None => remove link.
    def change_link(self, a: str, b: str, new_cost: Optional[int]):
        """
        Update both endpoints neighbor lists.
        """
        if a not in self.routers or b not in self.routers:
            raise ValueError("Both routers must exist")
        ra = self.routers[a]
        rb = self.routers[b]
        ra.update_link_cost(b, new_cost)
        rb.update_link_cost(a, new_cost)
        # store event in logs
        print(f"[SIM] tick {self.current_tick}: link {a}<->{b} changed to {new_cost}")

    # Pretty print last snapshot
    def print_routing_tables(self):
        print(f"--- Routing tables at tick {self.current_tick} ---")
        for rname, r in sorted(self.routers.items()):
            print(f"Router {rname}:")
            table = r.get_routing_table_snapshot()
            for dest in sorted(table.keys()):
                dist, next_hop, poisoned = table[dest]
                dist_str = f"{dist}" if dist < self.max_hop else "INF"
                print(f"  {dest:4} -> {dist_str:4} via {str(next_hop):6}{' [POISON]' if poisoned else ''}")
            print("")

    def dump_history(self):
        # For analysis: show how many ticks it took to settle per destination per router, etc.
        return copy.deepcopy(self.history)


# --- Demo main (runs when executed directly) ---
if __name__ == "__main__":
    def demo():
        print("Distance Vector Simulator — Level 3 Demo")
        sim = NetworkSimulator(periodic_interval=6, max_hop=16)

        # Set up a 4-node network
        topo = {
            "A": {"B": 1, "C": 4},
            "B": {"A": 1, "C": 2, "D": 5},
            "C": {"A": 4, "B": 2, "D": 1},
            "D": {"B": 5, "C": 1},
        }
        # Add routers with split horizon + poison reverse enabled (realistic)
        for name, nbrs in topo.items():
            sim.add_router(name, neighbors=nbrs, split_horizon=True, poison_reverse=True)

        # Run until convergence
        converged, tick = sim.run(max_ticks=200, until_converged=True, debug=True)
        print(f"Initial convergence: converged={converged}, tick={tick}")
        sim.print_routing_tables()

        # Now simulate a link cost change: B <-> D increases from 5 -> 20 (degradation)
        sim.change_link("B", "D", 20)
        # Run again to reach new convergence
        converged2, tick2 = sim.run(max_ticks=200, until_converged=True, debug=True)
        print(f"After change convergence: converged={converged2}, tick={tick2}")
        sim.print_routing_tables()

        # Now simulate a link failure: C <-> D goes down
        sim.change_link("C", "D", None)
        converged3, tick3 = sim.run(max_ticks=200, until_converged=True, debug=True)
        print(f"After failure convergence: converged={converged3}, tick={tick3}")
        sim.print_routing_tables()

        # Show brief convergence stats
        history = sim.dump_history()
        print(f"History length (ticks captured): {len(history)}")
        print("Demo complete.")

    demo()
