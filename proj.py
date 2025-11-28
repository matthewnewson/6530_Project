import pandas as pd
import numpy as np
import os

# --- 1. DATA LAYER (Robust CSV Loader) ---
class DataLoader:
    """
    Handles loading and sanitizing the specific Kaggle 'Cleaned_Laptop_data.csv'.
    """
    def __init__(self, filename='Cleaned_Laptop_data.csv'):
        self.filename = filename

    def clean_currency(self, x):
        """Converts '₹ 56,990' -> 56990.0"""
        if isinstance(x, (int, float)): return float(x)
        if pd.isna(x) or x == 'Missing': return 0.0
        # Remove currency symbols, commas, and whitespace
        clean_str = str(x).replace('₹', '').replace(',', '').strip()
        try:
            return float(clean_str)
        except ValueError:
            return 0.0

    def clean_capacity(self, x):
        """Converts '8 GB' -> 8, '512 GB' -> 512"""
        if isinstance(x, (int, float)): return float(x)
        if pd.isna(x) or x == 'Missing': return 0.0
        # Extract the first number found in the string
        clean_str = str(x).upper().replace('GB', '').replace('TB', '000').strip()
        try:
            return float(clean_str)
        except ValueError:
            return 0.0

    def load_data(self):

        print(f"Loading {self.filename}...")
        df = pd.read_csv(self.filename)
        
        # --- Specific Data Cleaning for this Kaggle Set ---
        
        # 1. Clean Price
        price_col = 'latest_price' if 'latest_price' in df.columns else 'price'
        df['clean_price'] = df[price_col].apply(self.clean_currency)
        
        # 2. Clean RAM
        df['clean_ram'] = df['ram_gb'].apply(self.clean_capacity)
        
        # 3. Clean Storage
        df['clean_ssd'] = df['ssd'].apply(self.clean_capacity)
        df['clean_hdd'] = df['hdd'].apply(self.clean_capacity)
        df['total_storage'] = df['clean_ssd'] + df['clean_hdd']
        
        # 4. Clean GPU/Performance
        if 'star_rating' in df.columns:
             df['quality_score'] = df['star_rating'].fillna(0)
        else:
             df['quality_score'] = df['clean_price'] / 10000 

        df['model'] = df['model'].fillna('Unknown Model')
        df = df.fillna(0)
        return df

    def get_mock_data(self):
        """Fallback if user hasn't downloaded the CSV yet"""
        data = {
            'model': [f'Laptop Model {i}' for i in range(1, 15)],
            'clean_price': np.random.randint(30000, 250000, 14),
            'clean_ram': [8, 16, 32, 4, 8, 16, 64, 8, 16, 32, 16, 8, 4, 32],
            'total_storage': [512, 1024, 512, 256, 512, 1024, 2048, 512, 512, 1024, 512, 256, 64, 2048],
            'quality_score': np.random.uniform(3.0, 5.0, 14),
            'ram_gb': ['8 GB', '16 GB', '32 GB', '4 GB', '8 GB', '16 GB', '64 GB', '8 GB', '16 GB', '32 GB', '16 GB', '8 GB', '4 GB', '32 GB']
        }
        return pd.DataFrame(data)

# --- 2. MODELING LAYER (Preferences P) ---
class PreferenceEngine:
    def __init__(self):
        self.profiles = {
            'gamer': {
                'weights': {'clean_price': -0.2, 'clean_ram': 0.4, 'total_storage': 0.2, 'quality_score': 0.2},
                'vocab': 'gamer',
                'desc': "Maximizing Specs and Performance."
            },
            'student': {
                'weights': {'clean_price': -0.8, 'clean_ram': 0.1, 'total_storage': 0.1, 'quality_score': 0.0},
                'vocab': 'student',
                'desc': "Maximizing Value and Affordability."
            },
            'professional': {
                'weights': {'clean_price': -0.3, 'clean_ram': 0.5, 'total_storage': 0.2, 'quality_score': 0.3},
                'vocab': 'pro',
                'desc': "Reliability and Multitasking capability."
            }
        }

    def get_profile(self, role_name):
        return self.profiles.get(role_name.lower())

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
                if max_val - min_val == 0:
                    normalized[col] = 0
                else:
                    normalized[col] = (df[col] - min_val) / (max_val - min_val)
        return normalized

    def semantic_translation(self, row, vocab_type):
        """Translates technical specs into user-friendly text."""
        
        # 1. Translate RAM
        ram_val = row['clean_ram']
        if vocab_type == 'gamer':
            if ram_val >= 16: ram_text = "Fastest"
            elif ram_val >= 8: ram_text = "Playable"
            else: ram_text = "Laggy"
        elif vocab_type == 'student':
            if ram_val >= 8: ram_text = "Fast"
            else: ram_text = "Slower"
        else: # pro
            ram_text = f"{int(ram_val)} GB DDR4/5"

        # 2. Translate Price
        price_val = row['clean_price']
        if vocab_type == 'student':
            if price_val > 80000: price_text = "Pricy"
            elif price_val > 40000: price_text = "Moderate Investment"
            else: price_text = "Great Deal"
        else:
            price_text = f"₹{price_val:,.0f}"

        return pd.Series([ram_text, price_text], index=['User_RAM', 'User_Price'])

    def transform(self, role_name, profile_data):
        weights = profile_data['weights']
        vocab = profile_data['vocab']
        
        # 1. Decision Phase
        norm_df = self.normalize_data(self.df)
        self.df['utility_score'] = 0
        
        for feature, weight in weights.items():
            if feature in norm_df.columns:
                val = norm_df[feature]
                if weight < 0:
                     score_contribution = abs(weight) * (1 - val)
                else:
                     score_contribution = weight * val
                self.df['utility_score'] += score_contribution

        # 2. Execution Phase
        translated = self.df.apply(lambda x: self.semantic_translation(x, vocab), axis=1)
        self.df = pd.concat([self.df, translated], axis=1)
        
        # Sort by score and take top 10 (UPDATED HERE)
        return self.df.sort_values(by='utility_score', ascending=False).head(10)

# --- 4. INTERFACE ---
def main():
    loader = DataLoader('Cleaned_Laptop_data.csv')
    raw_data = loader.load_data()
    
    prefs = PreferenceEngine()
    
    print(f"\n--- DATA WRANGLER (Dataset: {len(raw_data)} laptops) ---")
    
    while True:
        print("\nWho are you? (Type 'Gamer', 'Student', 'Professional') or 'q' to quit")
        role = input("> ").strip().lower()
        
        if role == 'q': break
        
        profile = prefs.get_profile(role)
        if not profile:
            print("Unknown role. Please try again.")
            continue
            
        print(f"\n[Analzying {len(raw_data)} laptops for a {role.upper()}...]")
        
        wrangler = DataWrangler(raw_data)
        results = wrangler.transform(role, profile)
        
        cols = ['model', 'User_Price', 'User_RAM', 'total_storage', 'utility_score']
        
        print(f"\n--- TOP 10 RECOMMENDATIONS FOR {role.upper()} ---")
        
        # UPDATED FORMATTING: Added col_space and justify parameters
        pd.set_option('display.max_colwidth', 40)
        print(results[cols].to_string(index=False, col_space=25, justify='left'))

if __name__ == "__main__":
    main()