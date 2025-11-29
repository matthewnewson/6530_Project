import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# ==========================================
#    LOGIC LAYER
# ==========================================

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
        if not os.path.exists(self.filename):
            return self.get_mock_data() 

        df = pd.read_csv(self.filename)
        price_col = 'latest_price' if 'latest_price' in df.columns else 'price'
        df['clean_price'] = df[price_col].apply(self.clean_currency)
        df['clean_ram'] = df['ram_gb'].apply(self.clean_capacity)
        df['clean_ssd'] = df['ssd'].apply(self.clean_capacity)
        df['clean_hdd'] = df['hdd'].apply(self.clean_capacity)
        df['total_storage'] = df['clean_ssd'] + df['clean_hdd']
        
        if 'star_rating' in df.columns:
             df['quality_score'] = df['star_rating'].fillna(0)
        else:
             df['quality_score'] = df['clean_price'] / 10000 

        df['model'] = df['model'].fillna('Unknown Model')
        df = df.fillna(0) 
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

class PreferenceEngine:
    def __init__(self):
        self.profiles = {
            'gamer': {'weights': {'clean_price': -0.2, 'clean_ram': 0.4, 'total_storage': 0.2, 'quality_score': 0.2}, 'vocab': 'gamer'},
            'student': {'weights': {'clean_price': -0.8, 'clean_ram': 0.1, 'total_storage': 0.1, 'quality_score': 0.0}, 'vocab': 'student'},
            'professional': {'weights': {'clean_price': -0.3, 'clean_ram': 0.5, 'total_storage': 0.2, 'quality_score': 0.3}, 'vocab': 'pro'}
        }
        self.current_weights = None

    def get_profile(self, role_name):
        p = self.profiles.get(role_name.lower())
        if p:
            self.current_weights = p['weights'].copy() 
            p['current_weights'] = self.current_weights
        return p

    def update_weight(self, feature, delta):
        if feature in self.current_weights:
            self.current_weights[feature] += delta

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
        ram_val = row['clean_ram']
        try: ram_int = int(ram_val)
        except: ram_int = 0
        
        if vocab_type == 'gamer': ram_text = "Fastest" if ram_val >= 16 else "Playable"
        elif vocab_type == 'student': ram_text = "Zoom Ready" if ram_val >= 8 else "Basic"
        else: ram_text = f"{ram_int} GB"

        price_val = row['clean_price']
        if vocab_type == 'student': price_text = "Expensive" if price_val > 60000 else "Deal"
        else: price_text = f"₹{price_val:,.0f}"

        store_val = row['total_storage']
        if store_val >= 1000: store_text = f"{store_val/1000:.1f} TB"
        else: store_text = f"{int(store_val)} GB"

        return pd.Series([ram_text, price_text, store_text], index=['User_RAM', 'User_Price', 'User_Storage'])

    def transform(self, role_name, profile_data):
        weights = profile_data['current_weights']
        vocab = profile_data['vocab']
        
        norm_df = self.normalize_data(self.df)
        self.df['utility_score'] = 0
        
        for feature, weight in weights.items():
            if feature in norm_df.columns:
                val = norm_df[feature].fillna(0)
                if weight < 0: score_contribution = abs(weight) * (1 - val)
                else: score_contribution = weight * val
                self.df['utility_score'] += score_contribution

        translated = self.df.apply(lambda x: self.semantic_translation(x, vocab), axis=1)
        final_df = pd.concat([self.df, translated], axis=1)
        return final_df.sort_values(by='utility_score', ascending=False)

# ==========================================
#    UI LAYER (Tkinter)
# ==========================================

class LaptopRecommenderUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Personalized Laptop Wrangler")
        self.root.geometry("950x600") # Made slightly wider to fit storage

        # 1. Init Data
        self.loader = DataLoader('Cleaned_Laptop_data.csv')
        self.raw_data = self.loader.load_data()
        self.prefs = PreferenceEngine()
        self.current_role = None
        self.current_profile = None
        
        # Default columns (Storage is now included by default)
        self.display_cols = ['id', 'model', 'User_Price', 'User_RAM', 'User_Storage', 'utility_score']

        # 2. Build UI Components
        self.setup_top_frame()
        self.setup_middle_frame()
        self.setup_bottom_frame()
        
        self.status_label.config(text=f"Database Loaded: {len(self.raw_data)} laptops ready.")

    def setup_top_frame(self):
        frame = tk.Frame(self.root, pady=10)
        frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Label(frame, text="Step 1: Select Your Role", font=("Arial", 12, "bold")).pack()
        
        btn_frame = tk.Frame(frame)
        btn_frame.pack(pady=5)
        
        # Role Buttons
        tk.Button(btn_frame, text="Gamer", width=15, command=lambda: self.select_role("gamer")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Student", width=15, command=lambda: self.select_role("student")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Professional", width=15, command=lambda: self.select_role("professional")).pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(frame, text="Welcome. Please select a role.", fg="blue")
        self.status_label.pack(pady=5)

    def setup_middle_frame(self):
        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack(expand=True, fill=tk.BOTH)
        
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Define Columns (Reordered: Storage before Score)
        columns = ("ID", "Model", "Price", "RAM", "Storage", "Score")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", yscrollcommand=scrollbar.set)
        
        # Format Columns
        self.tree.heading("ID", text="ID")
        self.tree.column("ID", width=40)
        self.tree.heading("Model", text="Model Name")
        self.tree.column("Model", width=280)
        self.tree.heading("Price", text="Price")
        self.tree.column("Price", width=100)
        self.tree.heading("RAM", text="RAM")
        self.tree.column("RAM", width=100)
        self.tree.heading("Storage", text="Storage")
        self.tree.column("Storage", width=100)
        self.tree.heading("Score", text="Util Score")
        self.tree.column("Score", width=80)

        self.tree.pack(expand=True, fill=tk.BOTH)
        scrollbar.config(command=self.tree.yview)

    def setup_bottom_frame(self):
        frame = tk.Frame(self.root, pady=15, bg="#f0f0f0")
        frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        tk.Label(frame, text="Step 2: Refine & Compare", bg="#f0f0f0", font=("Arial", 10, "bold")).pack()
        
        btn_frame = tk.Frame(frame, bg="#f0f0f0")
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="Price Matters More (+)", command=lambda: self.refine_weight('clean_price', -0.2)).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="RAM Matters More (+)", command=lambda: self.refine_weight('clean_ram', 0.2)).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Storage Matters More (+)", command=lambda: self.refine_weight('total_storage', 0.2)).pack(side=tk.LEFT, padx=5)
        
        tk.Frame(btn_frame, width=20, bg="#f0f0f0").pack(side=tk.LEFT) # Spacer
        
        tk.Button(btn_frame, text="Compare 2 items", bg="lightblue", command=self.compare_items).pack(side=tk.LEFT, padx=5)

    def select_role(self, role):
        self.current_role = role
        self.current_profile = self.prefs.get_profile(role)
        # Reset columns (Storage is always included)
        self.display_cols = ['id', 'model', 'User_Price', 'User_RAM', 'User_Storage', 'utility_score']
        self.update_table()
        self.status_label.config(text=f"Active Role: {role.upper()} - {self.current_profile['vocab']} view loaded.")

    def refine_weight(self, feature, delta):
        if not self.current_profile:
            messagebox.showwarning("No Role", "Please select a role first!")
            return
            
        self.prefs.update_weight(feature, delta)
        self.update_table()
        self.status_label.config(text=f"Updated weight for {feature}. Re-ranking...")

    def update_table(self):
        if not self.current_profile: return
        
        # 1. Run Data Wrangler
        wrangler = DataWrangler(self.raw_data)
        results = wrangler.transform(self.current_role, self.current_profile)
        top_20 = results.head(20)
        
        # 2. Clear Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # 3. Insert New Rows
        for _, row in top_20.iterrows():
            vals = [
                row['id'],
                row['model'],
                row['User_Price'],
                row['User_RAM'],
                row['User_Storage'], # Always included now
                f"{row['utility_score']:.2f}"
            ]
            self.tree.insert("", tk.END, values=vals)

    def compare_items(self):
        id_str = simpledialog.askstring("Compare", "Enter two Laptop IDs separated by comma (e.g. 5, 12):")
        if not id_str: return
        
        try:
            ids = [int(x.strip()) for x in id_str.split(',')]
            if len(ids) != 2: raise ValueError
            
            item1 = self.raw_data[self.raw_data['id'] == ids[0]].iloc[0]
            item2 = self.raw_data[self.raw_data['id'] == ids[1]].iloc[0]
            
            txt = f"--- COMPARISON ---\n\n"
            txt += f"Laptop A: {item1['model']}\nPrice: {item1['clean_price']}\nRAM: {item1['clean_ram']}\nStorage: {item1['total_storage']}\n\n"
            txt += f"Laptop B: {item2['model']}\nPrice: {item2['clean_price']}\nRAM: {item2['clean_ram']}\nStorage: {item2['total_storage']}"
            
            messagebox.showinfo("Comparison Result", txt)
            
        except:
            messagebox.showerror("Error", "Invalid Input. Please enter two IDs like: 1, 5")

if __name__ == "__main__":
    root = tk.Tk()
    app = LaptopRecommenderUI(root)
    root.mainloop()