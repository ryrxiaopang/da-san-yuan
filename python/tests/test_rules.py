"""Rule tests for the team's Singapore ruleset (see RULES.md).

If the team changes a rule, change RULES.md and the matching test together.
"""
import os

import pytest

from dasanyuan.bots import EfficiencyBot, HeuristicBot, RandomBot, Style
from dasanyuan.game import CLAIM, PASS, PONG, RON, Config, Game
from dasanyuan import obs
from dasanyuan.obs import View
from dasanyuan.scenario import load_dir
from dasanyuan.scoring import Meld, MeldKind, Pattern, WinContext, bonus_tai, payments, score_hand
from dasanyuan.selfplay import GameProgress, SelfPlayConfig, play_full_game, play_hand, verify_hand_invariants
from dasanyuan.shanten import shanten
from dasanyuan.tile import EAST, RED, WHITE, counts_of, flower_number, hand_string, is_animal, is_bonus, parse_tiles

SCENARIOS = os.path.join(os.path.dirname(__file__), "..", "..", "scenarios")


def t(s):
    return parse_tiles(s)


def tile(s):
    return t(s)[0]


def score(hand, win, melds, seat, prev, tsumo):
    """Score a hand string with exposed `melds`."""
    tiles = t(hand)
    bonus = [x for x in tiles if is_bonus(x)]
    ctx = WinContext(seat_wind=seat, prevailing_wind=prev, self_draw=tsumo, win_tile=tile(win))
    return score_hand(counts_of(tiles), melds, bonus, ctx)


def has(s, p):
    return any(x == p for x, _ in s.items)


def pong(s):
    return Meld(MeldKind.PONG, tile(s), None, 1)


def chow(s):
    return Meld(MeldKind.CHOW, tile(s), None, 3)


def bots_for_seed(seed, styles=None):
    styles = styles or Style.archetypes()
    return [HeuristicBot(styles[i], seed) for i in range(4)]


# ------------------------------------------------------------ tiles and shanten

def test_tile_roundtrip():
    x = t("123m456p789s1257z f1 g4 a2")
    assert len(x) == 16
    assert x[9] == EAST
    assert x[11] == WHITE
    assert x[12] == RED
    assert flower_number(x[13]) == 1
    assert flower_number(x[14]) == 4
    assert is_animal(x[15])
    assert hand_string(x) == "123m456p789s1257z f1 g4 a2"


def sh(s):
    return shanten(counts_of(t(s)), 4)


def test_shanten_complete_hands():
    assert sh("123m456p789s11122z") == -1
    assert sh("19m19p19s12345677z") == -1
    assert sh("11122233344455m") == -1


def test_shanten_ready_hands():
    assert sh("123m456p789s1112z") == 0
    assert sh("19m19p19s1234567z") == 0  # 13-sided orphans wait
    assert sh("1112345678999m") == 0     # nine gates shape


def test_shanten_far_hands():
    assert sh("147m258p369s1234z") >= 4


def test_shanten_exposed_melds():
    # one exposed meld: 10 concealed tiles, 3 sets + pair needed -> tenpai on tanki
    assert shanten(counts_of(t("123m456p789s1z")), 3) == 0


# ------------------------------------------------------------ tai patterns

def test_men_qing_is_one_tai():
    # Concealed, no other tai source: 123m 456p 789s 234s + 99m pair, closed wait on 3s
    s = score("123m456p789s234s99m", "3s", [], 1, 0, False)
    assert has(s, Pattern.MEN_QING)
    assert s.tai == 1


def test_open_hand_without_tai_is_zero():
    s = score("456p789s234s99m", "3s", [chow("1m")], 1, 0, False)
    assert s.tai == 0, s.items


def test_ping_hu_clean_is_four_tai():
    # all chows, plain pair, two-sided wait (won on 1s completing 123s from 23s)
    s = score("123m456p789s123s99m", "1s", [], 1, 0, False)
    assert has(s, Pattern.PING_HU)
    assert s.raw_tai == 5  # ping hu 4 + men qing 1


def test_ping_hu_needs_two_sided_wait():
    # 3s completes 1-2 edge -> not ping hu
    s = score("123m456p789s123s99m", "3s", [], 1, 0, False)
    assert not has(s, Pattern.PING_HU), s.items


def test_concealed_ping_hu_with_non_matching_flowers_keeps_four_tai():
    # seat South (1) -> matching flowers are f2/g2. f1 does not match.
    s = score("123m456p789s123s99m f1", "1s", [], 1, 0, False)
    assert has(s, Pattern.PING_HU)
    assert has(s, Pattern.MEN_QING)
    assert s.raw_tai == 5


def test_open_ping_hu():
    # exposed chow, no bonus tiles -> 4 tai, no men qing
    s = score("456p789s123s99m", "1s", [chow("1m")], 1, 0, False)
    assert has(s, Pattern.PING_HU)
    assert not has(s, Pattern.MEN_QING)
    assert s.raw_tai == 4
    # exposed chow with a non-matching flower -> 1 tai
    s = score("456p789s123s99m f1", "1s", [chow("1m")], 1, 0, False)
    assert has(s, Pattern.PING_HU_WITH_FLOWERS)
    assert s.raw_tai == 1


def test_ping_hu_void_with_scoring_bonus():
    s = score("123m456p789s123s99m f2", "1s", [], 1, 0, False)
    assert not has(s, Pattern.PING_HU) and not has(s, Pattern.PING_HU_WITH_FLOWERS)
    assert has(s, Pattern.SEAT_FLOWER)
    # animals also void it
    s = score("123m456p789s123s99m a3", "1s", [], 1, 0, False)
    assert not has(s, Pattern.PING_HU) and not has(s, Pattern.PING_HU_WITH_FLOWERS)


def test_ping_hu_pair_cannot_be_valued_honour():
    s = score("123m456p789s123s55z", "1s", [], 1, 0, False)
    assert not has(s, Pattern.PING_HU)


def test_full_flush_is_five_tai():
    s = score("11123455678999m", "5m", [], 1, 0, False)
    # 14 tiles needed; build a clean full flush
    s = s or score("1112345678999m5m", "5m", [], 1, 0, False)
    assert has(s, Pattern.FULL_FLUSH)
    assert s.tai == 5


def test_half_flush_two_tai():
    s = score("123m456m789m22m111z", "2m", [], 2, 0, False)
    # seat West(2), prevailing East(0): 111z East pong = prevailing wind
    assert has(s, Pattern.HALF_FLUSH)
    assert has(s, Pattern.PREVAILING_WIND_PONG)


def test_dragon_and_wind_pongs():
    s = score("123m456p789s22p", "2p", [pong("4p")], 0, 0, False)
    assert s.tai == 0, f"open hand with no tai source: {s.items}"
    s = score("123m456p22p111z", "2p", [pong("5z")], 0, 0, False)
    assert has(s, Pattern.DRAGON_PONG)
    assert has(s, Pattern.SEAT_WIND_PONG)
    assert has(s, Pattern.PREVAILING_WIND_PONG)
    assert s.raw_tai == 3


def test_all_pongs_two_tai():
    s = score("222m555p888s99s", "9s", [pong("4p")], 1, 0, False)
    assert has(s, Pattern.ALL_PONGS)


def test_small_three_dragons():
    s = score("555z666z77z123m", "7z", [pong("9p")], 1, 0, False)
    assert has(s, Pattern.SMALL_THREE_DRAGONS)
    assert not has(s, Pattern.DRAGON_PONG)


def test_limit_hands():
    s = score("19m19p19s12345677z", "7z", [], 1, 0, False)
    assert has(s, Pattern.THIRTEEN_ORPHANS)
    assert s.tai == 5
    s = score("555z666z777z123m99p", "9p", [], 1, 0, False)
    assert has(s, Pattern.BIG_THREE_DRAGONS)
    s = score("111z222z333z444z99p", "9p", [], 1, 0, True)
    assert has(s, Pattern.BIG_FOUR_WINDS)


def test_four_concealed_pongs_needs_self_draw_or_pair_wait():
    # completed pong by discard -> not four concealed
    s = score("222m555p888s999s11p", "9s", [], 1, 0, False)
    assert not has(s, Pattern.FOUR_CONCEALED_PONGS)
    s = score("222m555p888s999s11p", "9s", [], 1, 0, True)
    assert has(s, Pattern.FOUR_CONCEALED_PONGS)
    # pair wait on discard is fine
    s = score("222m555p888s999s11p", "1p", [], 1, 0, False)
    assert has(s, Pattern.FOUR_CONCEALED_PONGS)


def test_bonus_tiles():
    # seat East: f1 and g1 match; a1 animal
    tai, _ = bonus_tai(t("f1 g1 a1 f3"), 0)
    assert tai == 3
    tai, items = bonus_tai(t("f1 f2 f3 f4"), 2)
    assert tai == 2
    assert any(p == Pattern.FLOWER_SET for p, _ in items)


def test_tai_cap_is_five():
    s = score("1112345678999m5m f1 a1 a2", "5m", [], 0, 0, True)
    assert s.tai == 5
    assert s.raw_tai > 5


# ---------------------------------------------------------------- payments

def test_payments_match_team_table():
    # discarder pays for everyone: 4, 8, 16, 32, 64
    for tai, amt in [(1, 4), (2, 8), (3, 16), (4, 32), (5, 64)]:
        assert payments(tai, 0, 2) == [amt, 0, -amt, 0]
    # self draw: each pays 2, 4, 8, 16, 32
    for tai, each in [(1, 2), (2, 4), (3, 8), (4, 16), (5, 32)]:
        assert payments(tai, 1, None) == [-each, 3 * each, -each, -each]


# ----------------------------------------------------------- game engine

def test_deal_is_correct():
    g = Game(Config(), 42)
    assert g.players[0].hand_size() == 14
    for s in range(1, 4):
        assert g.players[s].hand_size() == 13
    bonus = sum(len(p.bonus) for p in g.players)
    drawn, held = g.tile_census()
    assert drawn == held
    assert drawn == 53 + bonus


def test_same_seed_same_game():
    def run(seed):
        _, r = play_hand(Game(Config(), seed), bots_for_seed(9))
        return r.winner, r.deltas, r.turns
    assert run(5) == run(5)


def test_thousands_of_hands_keep_invariants():
    styles = Style.archetypes()
    wins = 0
    for seed in range(3000):
        cfg = Config(dealer=seed % 4, prevailing_wind=(seed // 4) % 4)
        bots = [RandomBot(seed) if seed % 7 == 0 and i == 2 else HeuristicBot(styles[(i + seed) % 4], seed)
                for i in range(4)]
        checked = [0]

        def on_decision(g, seat, a):
            # every chosen action must be legal and the observation must encode
            assert a in g.legal_actions(seat)
            if checked[0] < 3:
                buf = obs.encode_obs(g, seat)
                assert sum(buf[obs.OFF_HAND:obs.OFF_HAND + 34]) == g.players[seat].hand_size()
                checked[0] += 1

        g, r = play_hand(Game(cfg, seed), bots, on_decision)
        verify_hand_invariants(g)
        if r.winner is not None:
            wins += 1
            assert r.score.tai >= 1
    assert wins > 1500, f"bots should win most hands, got {wins}"


def test_bots_only_see_public_info():
    # A bot's decision must not change if hidden information (other hands / wall) changes
    # while everything public stays the same. Swap two unseen tiles between opponents.
    g = Game(Config(), 77)
    g2 = g.clone()
    a = next(k for k in range(34) if g2.players[1].hand[k] > 0 and g2.players[2].hand[k] == 0)
    b = next(k for k in range(34) if g2.players[2].hand[k] > 0 and g2.players[1].hand[k] == 0)
    g2.players[1].hand[a] -= 1
    g2.players[1].hand[b] += 1
    g2.players[2].hand[b] -= 1
    g2.players[2].hand[a] += 1
    b1 = HeuristicBot(Style.balanced(), 3)
    b2 = HeuristicBot(Style.balanced(), 3)
    assert b1.act(View(g, 0)) == b2.act(View(g2, 0))
    assert obs.encode_obs(g, 0) == obs.encode_obs(g2, 0)


def test_claim_priority_ron_beats_pong():
    # Find a state where someone can ron and someone else can pong the same discard,
    # by brute force over seeds with heuristic play, and check the winner.
    for seed in range(4000):
        g = Game(Config(), seed)
        bots = bots_for_seed(seed)
        while (seat := g.to_act()) is not None:
            if g.phase.kind == CLAIM:
                ron = [s for s in range(4) if RON in g.legal_actions(s)]
                pongs = [s for s in range(4) if PONG in g.legal_actions(s)]
                if ron and any(p not in ron for p in pongs):
                    # everyone answers: ron seats ron, pong seats pong
                    while (s := g.to_act()) is not None:
                        g.apply(s, RON if s in ron else PONG if s in pongs else PASS)
                    r = g.result
                    assert r.winner is not None and r.winner in ron
                    return
            g.apply(seat, bots[seat].act(View(g, seat)))
    pytest.fail("no ron-vs-pong situation found in 4000 seeds")


# ------------------------------------------------- labels and scenario bank

def test_wait_labels_match_actual_wins():
    # Whenever someone wins on a discard, the label recorded at the discarder's decision
    # must say that tile wins for that opponent, for exactly that many tai.
    checked = 0
    for seed in range(1500):
        last = [None]

        def on_decision(g, seat, a):
            if a < 34:
                w, shs = obs.encode_labels(g, seat)
                # own shanten label equals a direct computation
                p = g.players[seat]
                assert shs[0] == max(-1, min(8, shanten(p.hand, p.melds_needed())))
                last[0] = (seat, a, w)

        _, r = play_hand(Game(Config(), seed), bots_for_seed(seed), on_decision)
        if r.winner is not None and r.discarder is not None and r.score is not None:
            seat, tl, labels = last[0]
            if seat == r.discarder and not has(r.score, Pattern.ROBBING_KONG):
                rel = (r.winner + 4 - r.discarder) % 4
                assert labels[(rel - 1) * 34 + tl] == r.score.tai, f"seed {seed}"
                checked += 1
    assert checked > 100, f"only {checked} discard wins checked"


def test_scenario_bank_is_valid_and_rules_hold():
    items = load_dir(SCENARIOS)
    assert len(items) >= 10
    for f, sc in items:
        sc.validate()
        # Rule scenarios test the engine itself, so any sensible policy must pass them.
        if "rules" in sc.tags:
            g = sc.build()
            err = sc.check(g, EfficiencyBot().act(View(g, sc.hero)))
            assert err is None, f"rule scenario {sc.name} failed: {err}"


# ------------------------------------------------------------ full games

def test_dealer_progression_follows_team_rules():
    start = GameProgress.start()
    # dealer wins: stays, repeat counter goes up
    p = start.next(True, False, False)
    assert (p.prevailing, p.dealer_no, p.repeat) == (0, 0, 1)
    # draw with no kong: dealer stays
    p = p.next(False, True, False)
    assert (p.prevailing, p.dealer_no, p.repeat) == (0, 0, 2)
    # draw where someone holds a kong: deal passes
    p = p.next(False, True, True)
    assert (p.prevailing, p.dealer_no, p.repeat) == (0, 1, 0)
    # another player wins: deal passes
    p = p.next(False, False, False)
    assert (p.dealer_no, p.repeat) == (2, 0)
    # after the fourth dealer the round wind moves on
    p = GameProgress(0, 3, 2).next(False, False, False)
    assert (p.prevailing, p.dealer_no, p.repeat) == (1, 0, 0)
    # the game ends after the North round's fourth dealer loses the deal
    assert GameProgress(3, 3, 0).next(False, False, False) is None
    # ...but not while the North dealer keeps winning
    assert GameProgress(3, 3, 0).next(True, False, False) is not None


def test_full_games_have_consistent_labels():
    cfg = SelfPlayConfig(hands=0, seed=3, styles=Style.archetypes(), random_lineup=True, shard_size=10,
                         out_dir="", compress=False, games=0)
    for game in range(20):
        labels = []

        def on_hand(_h, _s, hcfg, label, r, end):
            any_kong = any(m.is_kong() for p in end.players for m in p.melds)
            labels.append((hcfg.prevailing_wind, hcfg.dealer, label, r.winner, any_kong))

        play_full_game(cfg, game, None, on_hand)
        assert len(labels) >= 16, "a full game has at least 16 hands"
        for i in range(len(labels) - 1):
            pw, d, l, winner, kong = labels[i]
            pw2, d2, l2, _, _ = labels[i + 1]
            assert l.dealer_no == d + 1
            assert l2.hand_in_game == i + 2
            if winner == d or (winner is None and not kong):
                assert (pw2, d2, l2.repeat) == (pw, d, l.repeat + 1)
            else:
                assert l2.repeat == 0
                assert pw2 * 4 + d2 == pw * 4 + d + 1
        pw, d = labels[-1][0], labels[-1][1]
        assert (pw, d) == (3, 3), "game ends with the North round's fourth dealer"


def test_replay_trace_matches_recorded_game():
    """The replay viewer's game was recorded by the original engine; replaying it must be exact."""
    import json
    from dasanyuan.trace import trace_game
    path = os.path.join(os.path.dirname(__file__), "..", "..", "web", "replay", "replay.json")
    with open(path) as f:
        gold = json.load(f)
    run = gold["run"]
    cfg = SelfPlayConfig(hands=0, seed=run["seed"], random_lineup=run["random_lineup"],
                         shard_size=run["shard_size"])
    game_no = gold["games"][0]["game"] - 1
    hands = trace_game(cfg, game_no)
    assert json.loads(json.dumps(hands)) == gold["games"]
