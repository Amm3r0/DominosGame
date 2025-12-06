import pygame
import random
import sys
import os
import time
import copy

# --- إعدادات الألوان ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN_BG = (34, 139, 34)
YELLOW = (255, 215, 0)
RED = (200, 0, 0)
HIGHLIGHT = (0, 255, 0)

# --- إعدادات الشاشة ---
pygame.init()
info_obj = pygame.display.Info()
SCREEN_WIDTH = 1366
SCREEN_HEIGHT = 768
FPS = 60

BASE_PATH = r"D:\learn\Dominos Game\Media"

# مقاسات الكارت
CARD_W, CARD_H = 60, 120  
STOCK_AREA_WIDTH = 220     
GAP = 2 # مسافة صغيرة بين الكروت للتلامس

class Domino:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.is_double = (v1 == v2)
        self.image_orig = None 
        self.is_start = False 
        
    def get_value(self):
        return self.v1 + self.v2
        
    def __repr__(self):
        return f"[{self.v1}|{self.v2}]"
    
    def __eq__(self, other):
        if not isinstance(other, Domino): return False
        return (self.v1 == other.v1 and self.v2 == other.v2) or \
               (self.v1 == other.v2 and self.v2 == other.v1)

# ---------------------------------------------------------
# 🔥 الذكاء الاصطناعي 🔥
# ---------------------------------------------------------
class AdvancedAI:
    def __init__(self):
        self.opp_missed_numbers = set()
        self.my_hand = []
        self.start_time = 0
        self.TIME_LIMIT = 1.0 

    def reset(self):
        self.opp_missed_numbers = set()
        self.my_hand = []
        
    def record_pass(self, ends):
        if ends[0] is not None: self.opp_missed_numbers.add(ends[0])
        if ends[1] is not None: self.opp_missed_numbers.add(ends[1])

    def get_remaining_tiles(self, my_hand, board_cards):
        all_tiles = []
        for i in range(7):
            for j in range(i, 7):
                all_tiles.append((i, j))
        visible = set()
        for c in my_hand: visible.add(tuple(sorted((c.v1, c.v2))))
        for c in board_cards: visible.add(tuple(sorted((c.v1, c.v2))))
        unknowns = [t for t in all_tiles if tuple(sorted(t)) not in visible]
        return unknowns

    def get_valid_moves_sim(self, hand, ends):
        valid = []
        l, r = ends
        for i, card in enumerate(hand):
            v1, v2 = card 
            if l is None: 
                 valid.append({'idx': i, 'card': (v1,v2), 'side': 'left', 'new_l': v1, 'new_r': v2}) 
            else:
                if v1 == l: valid.append({'idx': i, 'card': (v1,v2), 'side': 'left', 'new_l': v2, 'new_r': r})
                elif v2 == l: valid.append({'idx': i, 'card': (v1,v2), 'side': 'left', 'new_l': v1, 'new_r': r})
                if (r != l or l is None):
                    if v1 == r: valid.append({'idx': i, 'card': (v1,v2), 'side': 'right', 'new_l': l, 'new_r': v2})
                    elif v2 == r: valid.append({'idx': i, 'card': (v1,v2), 'side': 'right', 'new_l': l, 'new_r': v1})
        valid.sort(key=lambda x: (x['card'][0] != x['card'][1]), reverse=False) 
        return valid

    def evaluate_board_state(self, my_hand_sim, opp_hand_guess, ends):
        score = 0
        opp_moves = self.get_valid_moves_sim(opp_hand_guess, ends)
        if not opp_moves and opp_hand_guess: score += 100000 
        my_suits = set()
        for c in my_hand_sim: my_suits.add(c[0]); my_suits.add(c[1])
        score += len(my_suits) * 50
        score += len(opp_hand_guess) * 500
        score -= len(my_hand_sim) * 500
        my_pts = sum([c[0]+c[1] for c in my_hand_sim])
        opp_pts = sum([c[0]+c[1] for c in opp_hand_guess])
        score += (opp_pts - my_pts) * 10
        return score

    def minimax(self, my_hand_sim, opp_hand_guess, ends, depth, is_maximizing, alpha, beta):
        if time.time() - self.start_time > self.TIME_LIMIT: raise TimeoutError 
        if depth == 0 or not my_hand_sim or not opp_hand_guess:
            return self.evaluate_board_state(my_hand_sim, opp_hand_guess, ends)

        if is_maximizing:
            max_eval = -float('inf')
            moves = self.get_valid_moves_sim(my_hand_sim, ends)
            if not moves: return self.minimax(my_hand_sim, opp_hand_guess, ends, depth-1, False, alpha, beta)
            for move in moves:
                new_hand = my_hand_sim[:]; new_hand.pop(move['idx'])
                new_ends = [move['new_l'], move['new_r']]
                eval_score = self.minimax(new_hand, opp_hand_guess, new_ends, depth-1, False, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha: break 
            return max_eval
        else:
            min_eval = float('inf')
            moves = self.get_valid_moves_sim(opp_hand_guess, ends)
            if not moves: return self.minimax(my_hand_sim, opp_hand_guess, ends, depth-1, True, alpha, beta)
            for move in moves:
                new_hand = opp_hand_guess[:]; new_hand.pop(move['idx'])
                new_ends = [move['new_l'], move['new_r']]
                eval_score = self.minimax(my_hand_sim, new_hand, new_ends, depth-1, True, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha: break 
            return min_eval

    def evaluate_hand(self, ai_hand_objects, board_ends, all_board_cards=[], forced_card=None):
        self.my_hand = ai_hand_objects
        self.start_time = time.time()
        
        if forced_card:
            for c in ai_hand_objects:
                if c == forced_card: return {'card': c, 'side': 'left'}

        my_hand_sim = [(c.v1, c.v2) for c in ai_hand_objects]
        unknowns = self.get_remaining_tiles(ai_hand_objects, all_board_cards)
        likely_opp_hand = [u for u in unknowns if u[0] not in self.opp_missed_numbers and u[1] not in self.opp_missed_numbers]
        if not likely_opp_hand: likely_opp_hand = unknowns
        import random
        random.shuffle(likely_opp_hand)
        opp_hand_guess = likely_opp_hand[:min(len(likely_opp_hand), 6)]

        possible_moves = self.get_valid_moves_sim(my_hand_sim, board_ends)
        if not possible_moves: return None
        if len(possible_moves) == 1:
            best_m = possible_moves[0]
            return {'card': ai_hand_objects[best_m['idx']], 'side': best_m['side']}

        best_move_global = possible_moves[0] 
        try:
            for depth in range(1, 20):
                best_val_this_depth = -float('inf')
                best_move_this_depth = None
                for move in possible_moves:
                    current_sim_hand = my_hand_sim[:]
                    current_sim_hand.pop(move['idx'])
                    current_ends = [move['new_l'], move['new_r']]
                    val = self.minimax(current_sim_hand, opp_hand_guess, current_ends, depth, False, -float('inf'), float('inf'))
                    if move['card'][0] == move['card'][1]: val += 20
                    if val > best_val_this_depth:
                        best_val_this_depth = val
                        best_move_this_depth = move
                if best_move_this_depth:
                    best_move_global = best_move_this_depth
        except TimeoutError: pass
        
        target_card_obj = ai_hand_objects[best_move_global['idx']]
        return {'card': target_card_obj, 'side': best_move_global['side']}

# ---------------------------------------------------------
# إدارة اللعبة
# ---------------------------------------------------------
class DominoGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Domino Master - Design Fixed")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22, bold=True)
        self.big_font = pygame.font.SysFont("Arial", 40, bold=True)
        
        self.assets = {}
        self.load_assets()
        self.ai = AdvancedAI()
        
        self.player_score = 0
        self.comp_score = 0
        self.match_target = 101
        self.round_num = 0
        self.last_winner = None 

        self.round_active = False
        self.wait_timer = 0 
        self.board = []
        self.ends = [None, None]
        self.player_hand = []
        self.comp_hand = []
        self.stock = []
        self.turn = 'player'
        
        self.forced_start_card = None 
        
        self.message = ""
        self.message_timer = 0
        
        self.selected_card = None
        self.valid_zones = []

    def load_assets(self):
        try:
            b_img = pygame.image.load(os.path.join(BASE_PATH, "Board_Game.png")).convert()
            self.assets['board'] = pygame.transform.smoothscale(b_img, (SCREEN_WIDTH, SCREEN_HEIGHT))
        except:
            self.assets['board'] = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)); self.assets['board'].fill(GREEN_BG)

        try:
            bk = pygame.image.load(os.path.join(BASE_PATH, "مقلوب.png")).convert_alpha()
            self.assets['back'] = pygame.transform.smoothscale(bk, (CARD_W, CARD_H))
        except:
            self.assets['back'] = pygame.Surface((CARD_W, CARD_H)); self.assets['back'].fill(BLACK)

        for i in range(7):
            for j in range(7):
                name = f"{i}_{j}.png"
                path = os.path.join(BASE_PATH, name)
                if not os.path.exists(path): path = os.path.join(BASE_PATH, f"{j}_{i}.png")
                
                if os.path.exists(path):
                    surf = pygame.image.load(path).convert_alpha()
                    surf = pygame.transform.smoothscale(surf, (CARD_W, CARD_H))
                    self.assets[f"{i}-{j}"] = surf
                else:
                    s = pygame.Surface((CARD_W, CARD_H)); s.fill(WHITE); pygame.draw.rect(s,BLACK,(0,0,CARD_W,CARD_H),2)
                    self.assets[f"{i}-{j}"] = s

    def start_new_round(self):
        self.round_num += 1
        all_doms = []
        for i in range(7):
            for j in range(i, 7):
                d = Domino(i, j)
                d.image_orig = self.assets.get(f"{i}-{j}", self.assets.get(f"{j}-{i}"))
                all_doms.append(d)
        
        random.shuffle(all_doms)
        self.player_hand = all_doms[:7]
        self.comp_hand = all_doms[7:14]
        self.stock = all_doms[14:]
        
        self.board = []
        self.ends = [None, None]
        self.ai.reset()
        self.round_active = True
        self.message = ""
        self.selected_card = None
        self.valid_zones = []
        self.forced_start_card = None
        self.determine_turn()

    def determine_turn(self):
        if self.round_num == 1:
            self.forced_start_card = None
            for i in range(6, -1, -1):
                for c in self.player_hand:
                    if c.v1 == i and c.v2 == i:
                        self.turn = 'player'
                        self.forced_start_card = c 
                        self.set_message(f"ابدأ بـ {c}")
                        return
                for c in self.comp_hand:
                    if c.v1 == i and c.v2 == i:
                        self.turn = 'comp'
                        self.forced_start_card = c 
                        return
            
            p_max_card = max(self.player_hand, key=lambda x: x.get_value())
            c_max_card = max(self.comp_hand, key=lambda x: x.get_value())
            
            if p_max_card.get_value() >= c_max_card.get_value():
                self.turn = 'player'
                self.forced_start_card = p_max_card
                self.set_message(f"ابدأ بـ {p_max_card}")
            else:
                self.turn = 'comp'
                self.forced_start_card = c_max_card
        else:
            self.forced_start_card = None
            if self.last_winner == 'player': self.turn = 'player'
            elif self.last_winner == 'comp': self.turn = 'comp'
            else: self.turn = 'player'

    def set_message(self, msg):
        self.message = msg
        self.message_timer = pygame.time.get_ticks()

    def play_card(self, card, side, player_key):
        hand = self.player_hand if player_key == 'player' else self.comp_hand
        if card in hand: hand.remove(card)
        
        if not self.board:
            card.is_start = True
            # البداية: اتجاه none، لكن الرسم هيتعامل معاه
            self.board.append({'card': card, 'connect_val': None, 'dir': 'none'}) 
            self.ends = [card.v1, card.v2]
        else:
            card.is_start = False
            target = self.ends[0] if side == 'left' else self.ends[1]
            new_val = card.v1 if card.v2 == target else card.v2
            matched_val = target
            
            if side == 'right':
                self.board.append({'card': card, 'connect_val': matched_val})
                self.ends[1] = new_val
            else: # Left
                self.board.insert(0, {'card': card, 'connect_val': matched_val})
                self.ends[0] = new_val

        self.selected_card = None
        self.valid_zones = []
        self.forced_start_card = None
        
        if len(hand) == 0: 
            self.end_round(player_key, "DOMINO!")
        else: 
            self.turn = 'comp' if player_key == 'player' else 'player'

    def is_blocked(self):
        if self.stock: return False
        return not (self.can_play(self.player_hand) or self.can_play(self.comp_hand))

    def can_play(self, hand):
        if not self.board: return True
        return any(d.v1 in self.ends or d.v2 in self.ends for d in hand)

    def handle_block(self):
        p_s = sum(d.get_value() for d in self.player_hand)
        c_s = sum(d.get_value() for d in self.comp_hand)
        
        if p_s < c_s: 
            self.end_round('player', "LOCKED! (You Won)", c_s)
        elif c_s < p_s: 
            self.end_round('comp', "LOCKED! (AI Won)", p_s)
        else: 
            self.end_round('draw', "DRAW!", 0)

    def end_round(self, winner, reason, points=None):
        self.round_active = False
        self.last_winner = winner
        pts = points
        if pts is None:
            opp = self.comp_hand if winner == 'player' else self.player_hand
            pts = sum(d.get_value() for d in opp)
        
        if winner == 'player':
            self.player_score += pts
            self.set_message(f"{reason} +{pts}")
        elif winner == 'comp':
            self.comp_score += pts
            self.set_message(f"{reason} +{pts}")
        else: self.set_message("Tie!")
        
        self.wait_timer = pygame.time.get_ticks()

    # --- خوارزمية الرسم المحسنة (Prevent Overlap) ---
    def calculate_layout(self):
        if not self.board:
            c = (SCREEN_WIDTH//2, SCREEN_HEIGHT//2)
            return [], c, c

        # 1. تحديد مركز البداية
        start_idx = 0
        for i, item in enumerate(self.board):
            if item['card'].is_start: start_idx = i; break
        
        layout_map = {}
        
        # الأبعاد
        S_LEN = CARD_H # طول الكارت العادي (120)
        S_WID = CARD_W # عرض الكارت العادي (60)
        
        # حدود الشاشة للكسر
        GRID_LIMIT_X = 450 
        
        # --- معالجة كارت البداية ---
        start_item = self.board[start_idx]
        s_card = start_item['card']
        
        # لو دوش: واقف (Angle 0). لو عادي: نايم (Angle 90).
        if s_card.is_double:
            start_angle = 0
            # في وضع 0: العرض 60 والطول 120
            w_start, h_start = S_WID, S_LEN 
        else:
            start_angle = 90
            # في وضع 90: العرض 120 والطول 60
            w_start, h_start = S_LEN, S_WID
            
        layout_map[start_idx] = {
            'x': 0, 'y': 0, 
            'w': w_start, 'h': h_start, 
            'angle': start_angle, 'item': start_item
        }

        # --- دالة مساعدة لحساب زاوية وأبعاد الكارت التالي ---
        def resolve_card_orientation(card, connect_val, direction):
            # direction: 1 (Right), -1 (Left)
            # return: angle, width_in_grid, height_in_grid
            
            if card.is_double:
                # الدوش دايما واقف (Angle 0) إلا لو حبينا نغيره، بس الستاندرد واقف
                return 0, S_WID, S_LEN 
            else:
                # العادي نايم (Angle 90 or 270)
                # حساب الزاوية عشان النقط تركب صح
                # لو ماشي يمين (1): لو v1 هو الـ connect يبقى v1 شمال، يبقى الزاوية 90 (v1 left, v2 right)
                if direction == 1:
                    ang = 90 if card.v1 == connect_val else 270
                else:
                    # لو ماشي شمال (-1): لو v1 هو الـ connect يبقى v1 يمين، يبقى الزاوية 270 (v1 right, v2 left)
                    ang = 270 if card.v1 == connect_val else 90
                return ang, S_LEN, S_WID

        # --- معالجة اليمين (Right Chain) ---
        # نبدأ من حافة كارت البداية اليمين
        current_x = w_start / 2 + GAP
        current_y = 0
        direction = 1 # 1: right, -1: left
        
        for i in range(start_idx + 1, len(self.board)):
            item = self.board[i]
            card = item['card']
            con = item['connect_val']
            
            # نحسب أبعاده المتوقعة لو كملنا في نفس الخط
            ang, w, h = resolve_card_orientation(card, con, direction)
            
            # هل وصلنا للحدود؟
            # نحسب الطرف البعيد للكارت
            far_edge = current_x + w if direction == 1 else current_x - w
            
            at_limit = (direction == 1 and far_edge > GRID_LIMIT_X) or \
                       (direction == -1 and far_edge < -GRID_LIMIT_X)
            
            if at_limit:
                # --- حالة الكسر (Turn) ---
                # الكارت ده هينزل تحت (Vertical turn)
                # دايما بنرسمه واقف (Angle 0/180) عشان الوصلة تبقى راسية
                
                # الزاوية الرأسية:
                # لو connect == v1، يبقى v1 فوق، الزاوية 0
                turn_angle = 0 if card.v1 == con else 180
                if card.is_double: turn_angle = 0
                
                turn_w, turn_h = S_WID, S_LEN # واقف (60, 120)
                
                # الموضع X:
                # لازم يفضل في نفس مكانه الأفقي (تحت الكارت اللي قبله)
                # بنرجعه شوية عشان يبقى سنتر تحت طرف اللي قبله
                # اللي قبله كان عند current_x (من الناحية اللي بنضيف منها)
                
                # بنثبت الـ X عند الطرف اللي وصلنا له
                # بس بنرحل نص عرض الكارت الجديد عشان السنتر
                if direction == 1: 
                    draw_x = current_x - (turn_w / 2) # محاولة محاذاة الطرف
                    # الأفضل: خليه يرجع لورا شوية عشان ميبقاش طاير
                    draw_x = current_x - GAP - turn_w/2 
                else:
                    draw_x = current_x + GAP + turn_w/2
                
                # الموضع Y:
                # ينزل تحت اللي قبله. اللي قبله كان على Y=current_y
                # وكان ارتفاعه (نص ارتفاعه الفعلي). الكارت الجديد هينزل
                # المسافة = نصف ارتفاع السابق + جاب + نصف ارتفاع الجديد
                # بما اننا مش عارفين ارتفاع السابق بدقة هنا (مخزناهوش)، هنفترض الاستاندرد (60)
                # وننزل مسافة آمنة
                drop_amount = (S_WID + S_LEN) / 2 + GAP + 10 # 90 + GAP
                # لكن السابق ممكن يكون دوش (120)، فالأمان نعتبر السابق كان كبير
                # هنزود drop ثابت
                step_down = S_LEN # 120
                
                draw_y = current_y + step_down
                
                # حفظ
                layout_map[i] = {'x': draw_x, 'y': draw_y, 'w': turn_w, 'h': turn_h, 'angle': turn_angle, 'item': item}
                
                # تحديث الحالة للي بعده
                current_y = draw_y # نزلنا تحت
                current_x = draw_x # الـ X الجديد هو سنتر الكارت ده
                
                # نعكس الاتجاه
                direction *= -1
                
                # نجهز الـ current_x للكارت الجاي
                # الكارت الجاي هيمشي عكس الاتجاه، هيبدأ من حافة الكارت ده
                if direction == 1: current_x += turn_w/2 + GAP
                else: current_x -= (turn_w/2 + GAP)
                
            else:
                # --- الوضع العادي (Straight) ---
                draw_x = current_x + (w/2) if direction == 1 else current_x - (w/2)
                draw_y = current_y
                
                layout_map[i] = {'x': draw_x, 'y': draw_y, 'w': w, 'h': h, 'angle': ang, 'item': item}
                
                # تحديث X للي بعده
                if direction == 1: current_x += w + GAP
                else: current_x -= (w + GAP)

        # --- معالجة اليسار (Left Chain) ---
        # نبدأ من حافة كارت البداية الشمال
        current_x = -w_start / 2 - GAP
        current_y = 0
        direction = -1 # start moving left
        
        for i in range(start_idx - 1, -1, -1):
            item = self.board[i]
            card = item['card']
            con = item['connect_val']
            
            ang, w, h = resolve_card_orientation(card, con, direction)
            
            far_edge = current_x - w if direction == -1 else current_x + w
            at_limit = (direction == -1 and far_edge < -GRID_LIMIT_X) or \
                       (direction == 1 and far_edge > GRID_LIMIT_X)
            
            if at_limit:
                # Turn Down (Vertical)
                turn_angle = 0 if card.v2 == con else 180 # Notice v2 here
                if card.is_double: turn_angle = 0
                turn_w, turn_h = S_WID, S_LEN
                
                if direction == -1:
                    draw_x = current_x + GAP + turn_w/2
                else:
                    draw_x = current_x - GAP - turn_w/2
                
                # Unified Look: Both chains drop DOWN (Snake style)
                # This prevents overlap with player hand which is at bottom, 
                # but since we center everything, it's safer to go UP for Left chain?
                # No, standard domino snake goes away from center or spirals.
                # Let's make Left chain go UP to balance the screen.
                
                step_up = S_LEN 
                draw_y = current_y - step_up # Go UP
                
                layout_map[i] = {'x': draw_x, 'y': draw_y, 'w': turn_w, 'h': turn_h, 'angle': turn_angle, 'item': item}
                
                current_y = draw_y
                current_x = draw_x
                direction *= -1
                
                if direction == -1: current_x -= (turn_w/2 + GAP)
                else: current_x += (turn_w/2 + GAP)
                
            else:
                draw_x = current_x - (w/2) if direction == -1 else current_x + (w/2)
                draw_y = current_y
                
                layout_map[i] = {'x': draw_x, 'y': draw_y, 'w': w, 'h': h, 'angle': ang, 'item': item}
                
                if direction == -1: current_x -= (w + GAP)
                else: current_x += (w + GAP)

        # --- تحويل للإحداثيات النهائية ---
        final_layout = []
        if not layout_map: return [], (0,0), (0,0)
        
        # Center everything on screen
        CENTER_X = (SCREEN_WIDTH - STOCK_AREA_WIDTH) // 2
        CENTER_Y = SCREEN_HEIGHT // 2
        
        l_tip, r_tip = (0,0), (0,0)
        
        sorted_keys = sorted(layout_map.keys())
        for idx in sorted_keys:
            entry = layout_map[idx]
            final_layout.append({
                'card': entry['item']['card'], 
                'pos': (entry['x'] + CENTER_X, entry['y'] + CENTER_Y), 
                'angle': entry['angle']
            })
            
        # Tips calculation (simplified)
        h_e = layout_map[sorted_keys[0]]
        hx, hy = h_e['x'] + CENTER_X, h_e['y'] + CENTER_Y
        l_tip = (hx, hy) # Approximated tip is center of last card

        t_e = layout_map[sorted_keys[-1]]
        tx, ty = t_e['x'] + CENTER_X, t_e['y'] + CENTER_Y
        r_tip = (tx, ty)
        
        return final_layout, l_tip, r_tip

    def run(self):
        self.start_new_round()
        running = True
        
        while running:
            if self.message and pygame.time.get_ticks() - self.message_timer > 2000:
                self.message = ""

            if not self.round_active:
                if pygame.time.get_ticks() - self.wait_timer > 3000:
                    if self.player_score < self.match_target and self.comp_score < self.match_target:
                        self.start_new_round()
            
            if self.round_active and self.is_blocked():
                self.handle_block()

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if self.player_score >= 101 or self.comp_score >= 101:
                        self.player_score = 0; self.comp_score = 0; self.round_num = 0; self.start_new_round()
                
                # --- Player Turn ---
                if self.round_active and self.turn == 'player' and event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    
                    # 1. Stock Interaction (تم الإصلاح)
                    stock_zone = pygame.Rect(SCREEN_WIDTH - 150, 100, 100, 500)
                    if stock_zone.collidepoint(mx, my):
                         # ممنوع السحب لو معاك لعب
                         if self.can_play(self.player_hand):
                             self.set_message("العب من ورقك!")
                         # ممنوع السحب في اول دور
                         elif self.forced_start_card:
                             self.set_message("لازم تلعب الأول!")
                         else:
                             if self.stock:
                                 card = self.stock.pop()
                                 # Auto-Play Check
                                 l, r = self.ends
                                 fits_l = (not self.board) or (card.v1 == l or card.v2 == l)
                                 fits_r = self.board and (card.v1 == r or card.v2 == r)
                                 
                                 if fits_l or fits_r: 
                                     side = 'left' if fits_l else 'right'
                                     self.play_card(card, side, 'player')
                                     self.set_message("سحبت ولعبت!")
                                 else:
                                     self.player_hand.append(card)
                                     self.set_message("سحبت (مش لايقة)")
                                     if not self.stock and not self.can_play(self.player_hand):
                                         self.set_message("باص (السحب خلص)")
                                         self.ai.record_pass(self.ends)
                                         self.turn = 'comp'
                             else:
                                 if not self.can_play(self.player_hand):
                                     self.set_message("باص!")
                                     self.turn = 'comp'
                                     self.ai.record_pass(self.ends)

                    # 2. Hand Interaction
                    hand_len = len(self.player_hand)
                    h_start_x = (SCREEN_WIDTH - (hand_len*(CARD_W+10))) // 2
                    for i, c in enumerate(self.player_hand):
                        r = pygame.Rect(h_start_x + i*(CARD_W+10), SCREEN_HEIGHT - CARD_H - 20, CARD_W, CARD_H)
                        if r.collidepoint(mx, my):
                            # فرض الكارت الاجباري
                            if self.forced_start_card and c != self.forced_start_card:
                                self.set_message(f"لازم تلعب {self.forced_start_card}")
                                continue

                            if self.selected_card == c: self.selected_card = None; self.valid_zones = []
                            else:
                                self.selected_card = c; self.valid_zones = []
                                _, l_tip, r_tip = self.calculate_layout()
                                l, r = self.ends
                                if not self.board or c.v1 == l or c.v2 == l:
                                    self.valid_zones.append({'rect': pygame.Rect(l_tip[0]-40, l_tip[1]-40, 80, 80), 'side': 'left'})
                                if self.board and (c.v1 == r or c.v2 == r) and (r != l or not self.board):
                                    self.valid_zones.append({'rect': pygame.Rect(r_tip[0]-40, r_tip[1]-40, 80, 80), 'side': 'right'})

                    if self.selected_card:
                        for zone in self.valid_zones:
                            if zone['rect'].collidepoint(mx, my):
                                self.play_card(self.selected_card, zone['side'], 'player')
                                break

            # --- AI Turn ---
            if self.round_active and self.turn == 'comp':
                self.draw(self.screen); pygame.display.flip(); time.sleep(0.5)
                
                board_cards_list = [item['card'] for item in self.board]
                move = self.ai.evaluate_hand(self.comp_hand, self.ends, board_cards_list, self.forced_start_card)
                
                if move:
                    self.play_card(move['card'], move['side'], 'comp')
                else:
                    if self.stock:
                        c = self.stock.pop()
                        l, r = self.ends
                        if not self.board or c.v1 == l or c.v2 == l: 
                            self.play_card(c, 'left', 'comp')
                        elif c.v1 == r or c.v2 == r: 
                            self.play_card(c, 'right', 'comp')
                        else: 
                            self.comp_hand.append(c)
                            self.set_message("AI Draws")
                            self.draw(self.screen); pygame.display.flip(); time.sleep(0.3)
                            
                            if not self.stock and not self.can_play(self.comp_hand):
                                self.set_message("AI PASS")
                                self.ai.record_pass(self.ends) 
                                self.turn = 'player'
                    else:
                        self.set_message("AI PASS")
                        self.ai.record_pass(self.ends) 
                        self.turn = 'player'

            self.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(FPS)
            
    def draw(self, screen):
        screen.blit(self.assets['board'], (0,0))
        layout, l_tip, r_tip = self.calculate_layout()
        
        for item in layout:
            img = item['card'].image_orig
            final_img = pygame.transform.rotate(img, item['angle'])
            rect = final_img.get_rect(center=item['pos'])
            screen.blit(final_img, rect)

        for zone in self.valid_zones:
            pygame.draw.circle(screen, HIGHLIGHT, zone['rect'].center, 15)
            pygame.draw.circle(screen, BLACK, zone['rect'].center, 15, 2)

        h_len = len(self.player_hand)
        h_start = (SCREEN_WIDTH - (h_len*(CARD_W+10))) // 2
        for i, c in enumerate(self.player_hand):
            x, y = h_start + i*(CARD_W+10), SCREEN_HEIGHT - CARD_H - 20
            if c == self.selected_card: y -= 20
            if c == self.forced_start_card: y -= 10
            
            screen.blit(c.image_orig, (x, y))
            
            if c == self.forced_start_card:
                pygame.draw.circle(screen, RED, (x+CARD_W//2, y-10), 5)

        c_len = len(self.comp_hand)
        c_start = (SCREEN_WIDTH - (c_len*(CARD_W+5))) // 2
        for i in range(c_len): screen.blit(self.assets['back'], (c_start + i*(CARD_W+5), 20))

        sx, sy = SCREEN_WIDTH - 120, 100
        if self.stock:
            for i in range(min(len(self.stock), 5)): screen.blit(self.assets['back'], (sx, sy + i*5))
            txt = self.font.render(f"DRAW ({len(self.stock)})", True, WHITE)
            screen.blit(txt, (sx, sy + CARD_H + 10))
        else:
            txt = self.font.render("Empty", True, (150,150,150)); screen.blit(txt, (sx, sy))

        pygame.draw.rect(screen, BLACK, (SCREEN_WIDTH//2 - 200, 0, 400, 40))
        s_txt = self.font.render(f"YOU: {self.player_score}  |  PC: {self.comp_score}  (Round {self.round_num})", True, YELLOW)
        screen.blit(s_txt, (SCREEN_WIDTH//2 - 150, 10))

        if self.message:
            m_surf = self.big_font.render(self.message, True, WHITE)
            m_rect = m_surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            pygame.draw.rect(screen, RED if "Win" not in self.message else (0,100,0), m_rect.inflate(20,20))
            screen.blit(m_surf, m_rect)

        if self.player_score >= 101 or self.comp_score >= 101:
            win_txt = "YOU WON!" if self.player_score >= 101 else "PC WON!"
            surf = self.big_font.render(win_txt, True, YELLOW)
            rect = surf.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 80))
            pygame.draw.rect(screen, BLACK, rect.inflate(20,20))
            screen.blit(surf, rect)

if __name__ == "__main__":
    game = DominoGame()
    game.run()