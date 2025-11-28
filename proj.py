import pandas as pd
import numpy as np
import os
import sys

# --- 1. DATA LAYER ---
class DataLoader:
    def __init__(self, filename='Cleaned_Laptop_data.csv'):
        self.filename = filename

    def clean_currency(self, x):
        if isinstance(x, (int, float)): return float(x)
        if pd.isna(x) or x == 'Missing': return 0.0
        clean_str = str(x).replace('₹', '').replace(',', '').strip()
        try: return float(clean_str)
        except ValueError: return 0.0

    def clean_capacity(self, x):
        if isinstance(x, (int, float)): return float(x)
        if pd.isna(x) or x == 'Missing': return 0.0
        clean_str = str(x).upper().replace('GB', '').replace('TB', '000').strip()
        try: return float(clean_str)
        except ValueError: return 0.0

    def load_data(self):
        print(f"Loading {self.filename}...")
        if not os.path.exists(self.filename):
            print("File not found. Using Mock Data.")
            return self.get_mock_data()

        df = pd.read_csv(self.filename)
        
        price_col = 'latest_price' if 'latest_price' in df.columns else 'price'
        df['clean_price'] = df[price_col].apply(self.clean_currency)
        df['clean_ram'] = df['ram_gb'].apply(self.clean_capacity)
        df['clean_ssd'] = df['ssd'].apply(self.clean_capacity)
        df['clean_hdd'] = df['hdd'].apply(self.clean_capacity)
        df['total_storage'] = df['clean_ssd'] + df['clean_hdd']
        
        # Synthetic Quality Score
        if 'star_rating' in df.columns:
             df['quality_score'] = df['star_rating'].fillna(0)
        else:
             df['quality_score'] = df['clean_price'] / 10000 

        df['model'] = df['model'].fillna('Unknown Model')
        df = df.fillna(0) 
        
        # Create a unique ID for selection later
        df['id'] = range(1, len(df) + 1)
        return df

    def get_mock_data(self):
        data = {
            'model': [f'Laptop {i}' for i in range(1, 20)],
            'clean_price': np.random.randint(30000, 250000, 19),
            'clean_ram': np.random.choice([8, 16, 32], 19),
            'total_storage': np.random.choice([256, 512, 1024], 19),
            'quality_score': np.random.uniform(3.0, 5.0, 19),
            'ram_gb': ['16 GB'] * 19 
        }
        df = pd.DataFrame(data)
        df['id'] = range(1, len(df) + 1)
        return df

# --- 2. MODELING LAYER ---
class PreferenceEngine:
    def __init__(self):
        self.profiles = {
            'gamer': {
                # High positive for RAM/Storage, Negative for Price
                'weights': {'clean_price': -0.2, 'clean_ram': 0.4, 'total_storage': 0.2, 'quality_score': 0.2},
                'vocab': 'gamer'
            },
            'student': {
                # Very high negative for Price (Cost sensitive)
                'weights': {'clean_price': -0.8, 'clean_ram': 0.1, 'total_storage': 0.1, 'quality_score': 0.0},
                'vocab': 'student'
            },
            'professional': {
                # Balanced
                'weights': {'clean_price': -0.3, 'clean_ram': 0.5, 'total_storage': 0.2, 'quality_score': 0.3},
                'vocab': 'pro'
            }
        }
        self.current_weights = None

    def get_profile(self, role_name):
        p = self.profiles.get(role_name.lower())
        if p:
            # Deep copy weights so we can modify them dynamically
            self.current_weights = p['weights'].copy() 
            p['current_weights'] = self.current_weights
        return p

    def update_weight(self, feature, delta):
        """Dynamic Learning: Adjusts weight based on user feedback"""
        if feature in self.current_weights:
            self.current_weights[feature] += delta
            print(f"   [Learning] Updated importance of {feature}: {self.current_weights[feature]:.2f}")

# --- 3. DECISION & EXECUTION LAYER ---
class DataWrangler:
    def __init__(self, df):
        self.df = df.copy()

    def normalize_data(self, df):
        numeric_cols = ['clean_price', 'clean_ram', 'total_storage', 'quality_score']
        normalized = df.copy()
        for col in numeric_cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val - min_val == 0: normalized[col] = 0
                else: normalized[col] = (df[col] - min_val) / (max_val - min_val)
        return normalized

    def semantic_translation(self, row, vocab_type):
        # 1. RAM
        ram_val = row['clean_ram']
        try: ram_int = int(ram_val)
        except: ram_int = 0
        
        if vocab_type == 'gamer': ram_text = "Fastest" if ram_val >= 16 else "Playable"
        elif vocab_type == 'student': ram_text = "Zoom Ready" if ram_val >= 8 else "Basic"
        else: ram_text = f"{ram_int} GB"

        # 2. PRICE
        price_val = row['clean_price']
        if vocab_type == 'student': price_text = "Expensive" if price_val > 60000 else "Deal"
        else: price_text = f"₹{price_val:,.0f}"

        # 3. STORAGE
        store_val = row['total_storage']
        if store_val >= 1000: store_text = f"{store_val/1000:.1f} TB"
        else: store_text = f"{int(store_val)} GB"

        return pd.Series([ram_text, price_text, store_text], 
                         index=['User_RAM', 'User_Price', 'User_Storage'])

    def transform(self, role_name, profile_data):
        weights = profile_data['current_weights']
        vocab = profile_data['vocab']
        
        # 1. SCORING
        # Apply Multi-Attribute Utility Theory (MAUT)
        norm_df = self.normalize_data(self.df)
        self.df['utility_score'] = 0
        
        for feature, weight in weights.items():
            if feature in norm_df.columns:
                val = norm_df[feature].fillna(0)
                if weight < 0: 
                    # For costs (negative weight), minimize value
                    score_contribution = abs(weight) * (1 - val)
                else: 
                    # For benefits (positive weight), maximize value
                    score_contribution = weight * val
                self.df['utility_score'] += score_contribution

        # 2. TRANSLATION (View Layer)
        translated = self.df.apply(lambda x: self.semantic_translation(x, vocab), axis=1)
        final_df = pd.concat([self.df, translated], axis=1)
        
        # 3. SORTING
        return final_df.sort_values(by='utility_score', ascending=False)

    def compare_items(self, id1, id2):
        """Side-by-side comparison for Decision Support"""
        item1 = self.df[self.df['id'] == id1].iloc[0]
        item2 = self.df[self.df['id'] == id2].iloc[0]
        
        print(f"\n--- COMPARISON: #{id1} vs #{id2} ---")
        cols = ['model', 'clean_price', 'clean_ram', 'total_storage', 'quality_score']
        comp_df = pd.DataFrame([item1[cols], item2[cols]], index=[f"Laptop {id1}", f"Laptop {id2}"])
        print(comp_df.T)

# --- 4. INTERFACE ---
def main():
    loader = DataLoader('Cleaned_Laptop_data.csv')
    raw_data = loader.load_data()
    prefs = PreferenceEngine()
    
    print(f"\n--- PERSONALIZED DATA WRANGLER (Dataset: {len(raw_data)} items) ---")
    
    while True:
        role = input("\nRole (Gamer/Student/Professional) or 'q': ").strip().lower()
        if role == 'q': break
        
        profile = prefs.get_profile(role)
        if not profile: continue
        
        # Default Columns
        current_cols = ['id', 'model', 'User_Price', 'User_RAM', 'utility_score']
            
        # LOOP FOR QUERY REFINEMENT
        while True:
            wrangler = DataWrangler(raw_data)
            results = wrangler.transform(role, profile)
            
            print(f"\n--- TOP 10 RECOMMENDATIONS (Ranked by Utility) ---")
            pd.set_option('display.max_colwidth', 30)
            
            # Use the dynamic 'current_cols' list
            print(results[current_cols].head(10).to_string(index=False, col_space=15, justify='left'))
            
            print("\nACTIONS: [C]ompare IDs | [R]efine Preferences | [N]ew Role | [Q]uit")
            action = input("> ").strip().lower()
            
            if action == 'n': break 
            elif action == 'q': return 
            elif action == 'c':
                try:
                    id1 = int(input("Enter ID 1: "))
                    id2 = int(input("Enter ID 2: "))
                    wrangler.compare_items(id1, id2)
                except: print("Invalid IDs")
                
            elif action == 'r':
                print("Feedback Loop: Which feature matters MORE to you right now?")
                print("1. Price  2. RAM  3. Storage")
                choice = input("Choice: ")
                
                if choice == '1': 
                    prefs.update_weight('clean_price', -0.2) 
                elif choice == '2': 
                    prefs.update_weight('clean_ram', 0.2)
                elif choice == '3': 
                    prefs.update_weight('total_storage', 0.2)
                    
                    # DYNAMIC UI UPDATE:
                    # If user cares about Storage, add it to the display if not already there
                    if 'User_Storage' not in current_cols:
                        # Insert it before utility_score
                        current_cols.insert(4, 'User_Storage')
                        print("   [UI] 'Storage' column added to your view.")
                        
                print("Re-calculating rankings...")

if __name__ == "__main__":
    main()