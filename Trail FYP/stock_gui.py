import tkinter as tk
from tkinter import ttk, messagebox
import requests
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class StockGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Market Prediction System")
        self.root.geometry("1200x800")
        
        # API configuration
        self.api_key = "ZTJL216JZCLZE26O"
        self.base_url = "http://127.0.0.1:8000"
        self.current_symbol = None
        
        # Create main frames
        self.create_frames()
        self.create_widgets()
        
        # Start periodic updates
        self.update_data()
        
    def create_frames(self):
        # Create main container frame
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create sidebar frame
        sidebar_frame = ttk.LabelFrame(container, text="Popular Companies & Actions", padding="10")
        sidebar_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        # Create company list frame
        self.company_frame = ttk.Frame(sidebar_frame)
        self.company_frame.pack(fill="y", expand=True)
        
        # Create action buttons frame
        action_frame = ttk.LabelFrame(sidebar_frame, text="Actions", padding="10")
        action_frame.pack(fill="x", padx=5, pady=5)
        
        # Create main content frame
        content_frame = ttk.Frame(container)
        content_frame.pack(side="right", fill="both", expand=True)
        
        # Search frame
        self.search_frame = ttk.LabelFrame(content_frame, text="Company Search", padding="10")
        self.search_frame.pack(fill="x", padx=5, pady=5)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(content_frame, text="Search Results", padding="10")
        self.results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Stock details frame
        self.stock_frame = ttk.LabelFrame(content_frame, text="Stock Details", padding="10")
        self.stock_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add popular companies
        self.add_popular_companies()
        
        # Add action buttons
        self.add_action_buttons(action_frame)
        
    def add_popular_companies(self):
        """
        Add a list of popular companies to the sidebar
        """
        popular_companies = [
            {"symbol": "AAPL", "name": "Apple Inc"},
            {"symbol": "GOOGL", "name": "Alphabet Inc"},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "AMZN", "name": "Amazon.com, Inc."},
            {"symbol": "META", "name": "Meta Platforms Inc"},
            {"symbol": "TSLA", "name": "Tesla, Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
            {"symbol": "JNJ", "name": "Johnson & Johnson"},
            {"symbol": "V", "name": "Visa Inc"},
            {"symbol": "DIS", "name": "The Walt Disney Company"},
            {"symbol": "BABA", "name": "Alibaba Group Holding Limited"}
        ]
        
        # Create a scrollable frame for the company list
        canvas = tk.Canvas(self.company_frame)
        scrollbar = ttk.Scrollbar(self.company_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add company buttons
        for company in popular_companies:
            btn = ttk.Button(
                scrollable_frame,
                text=f"{company['symbol']} - {company['name']}",
                command=lambda c=company: self.select_company(c)
            )
            btn.pack(fill="x", pady=2)
            
    def add_action_buttons(self, frame):
        """
        Add action buttons to the sidebar
        """
        ttk.Button(frame, text="Add New Company", command=self.add_new_company).pack(fill="x", pady=2)
        ttk.Button(frame, text="Refresh Data", command=self.refresh_data).pack(fill="x", pady=2)
        
    def select_company(self, company):
        """
        Handle company selection from the sidebar
        """
        self.current_symbol = company['symbol']
        
        # Update company details
        self.symbol_label.config(text=f"Symbol: {company['symbol']}")
        self.name_label.config(text=f"Name: {company['name']}")
        
        # Fetch current price
        try:
            url = f"{self.base_url}/predict-stock"
            response = requests.post(url, json={"symbol": company['symbol']})
            response.raise_for_status()
            result = response.json()
            
            self.price_label.config(text=f"Current Price: ${result['predicted_price']:.2f}")
            self.trend_label.config(text=f"Trend: {result['trend_prediction']} ({result['confidence']:.2f}% confidence)")
            
            # Update chart
            self.update_chart(company['symbol'])
            
        except requests.exceptions.HTTPError as http_err:
            messagebox.showerror("Error", f"HTTP error occurred: {http_err}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch stock details: {str(e)}")

    def create_widgets(self):
        # Search widgets
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(self.search_frame, textvariable=self.search_var, width=50)
        search_entry.pack(side="left", padx=5, pady=5)
        
        search_button = ttk.Button(self.search_frame, text="Search", command=self.search_companies)
        search_button.pack(side="left", padx=5, pady=5)
        
        # Results treeview
        self.results_tree = ttk.Treeview(self.results_frame, columns=("Symbol", "Name", "Type", "Region", "Score"), show="headings")
        self.results_tree.heading("Symbol", text="Symbol")
        self.results_tree.heading("Name", text="Name")
        self.results_tree.heading("Type", text="Type")
        self.results_tree.heading("Region", text="Region")
        self.results_tree.heading("Score", text="Score")
        
        self.results_tree.column("Symbol", width=100)
        self.results_tree.column("Name", width=300)
        self.results_tree.column("Type", width=100)
        self.results_tree.column("Region", width=150)
        self.results_tree.column("Score", width=100)
        
        self.results_tree.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bind double-click event
        self.results_tree.bind("<<TreeviewSelect>>", self.on_company_select)
        
        # Loading indicator
        self.loading_label = ttk.Label(self.results_frame, text="Searching...")
        self.loading_label.pack(pady=5)
        self.loading_label.pack_forget()  # Hide initially
        
        # Stock details widgets
        self.symbol_label = ttk.Label(self.stock_frame, text="Symbol:")
        self.symbol_label.pack(anchor="w", padx=5, pady=2)
        
        self.name_label = ttk.Label(self.stock_frame, text="Name:")
        self.name_label.pack(anchor="w", padx=5, pady=2)
        
        self.price_label = ttk.Label(self.stock_frame, text="Current Price:")
        self.price_label.pack(anchor="w", padx=5, pady=2)
        
        self.trend_label = ttk.Label(self.stock_frame, text="Trend:")
        self.trend_label.pack(anchor="w", padx=5, pady=2)
        
        self.predict_button = ttk.Button(self.stock_frame, text="Predict Price", command=self.predict_price)
        self.predict_button.pack(pady=10)
        
        # Create a figure for plotting
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.stock_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
    def search_companies(self):
        """
        Search for companies based on the search query
        """
        query = self.search_var.get().strip()
        if not query:
            return
            
        try:
            self.loading_label.pack(pady=5)
            self.root.update()
            
            # Make GET request to search endpoint
            response = requests.get(f"{self.base_url}/search-company?query={query}")
            response.raise_for_status()
            results = response.json()
            
            # Clear existing results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Add new results
            for company in results:
                self.results_tree.insert("", "end", values=(
                    company['symbol'],
                    company['name'],
                    company['type'],
                    company['region'],
                    company['matchScore']
                ))
            
            self.loading_label.pack_forget()
            
        except Exception as e:
            self.loading_label.pack_forget()
            messagebox.showerror("Error", f"Failed to search companies: {str(e)}")
            
    def on_company_select(self, event):
        selected_item = self.results_tree.selection()
        if not selected_item:
            return
            
        item = self.results_tree.item(selected_item[0])
        symbol = item['values'][0]
        self.current_symbol = symbol
        
        # Update company details
        self.symbol_label.config(text=f"Symbol: {symbol}")
        self.name_label.config(text=f"Name: {item['values'][1]}")
        
        # Fetch current price and trend
        try:
            # Make GET request to predict endpoint
            response = requests.get(f"{self.base_url}/predict-stock?symbol={symbol}")
            response.raise_for_status()
            result = response.json()
            
            self.price_label.config(text=f"Current Price: ${result['predicted_price']:.2f}")
            self.trend_label.config(text=f"Trend: {result['trend_prediction']} ({result['confidence']:.2f}% confidence)")
            
            # Update chart
            self.update_chart(symbol)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch stock details: {str(e)}")

    def predict_price(self):
        selected_item = self.results_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a company first")
            return
            
        item = self.results_tree.item(selected_item[0])
        symbol = item['values'][0]
        
        try:
            # Make GET request to predict endpoint
            response = requests.get(f"{self.base_url}/predict-stock?symbol={symbol}")
            response.raise_for_status()
            result = response.json()
            
            messagebox.showinfo("Prediction", 
                f"Predicted Price for {symbol}: ${result['predicted_price']:.2f}\n"
                f"Prediction Date: {result['prediction_date']}"
            )
            
            # Update chart
            self.update_chart(symbol)
            
        except requests.exceptions.HTTPError as http_err:
            messagebox.showerror("Error", f"HTTP error occurred: {http_err}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict price: {str(e)}")
            
    def update_chart(self, symbol):
        try:
            # Fetch historical data
            response = requests.get(f"{self.base_url}/predict-stock?symbol={symbol}")
            response.raise_for_status()
            result = response.json()
            
            # Clear previous plot
            self.ax.clear()
            
            # Plot historical data
            dates = [datetime.strptime(date, "%Y-%m-%d") for date in result['historical_dates']]
            prices = result['historical_prices']
            
            self.ax.plot(dates, prices, label="Historical Prices")
            
            # Add predicted price
            if 'prediction_date' in result:
                pred_date = datetime.strptime(result['prediction_date'], "%Y-%m-%d")
                self.ax.plot(pred_date, result['predicted_price'], 'ro', label="Predicted Price")
            
            self.ax.set_title(f"{symbol} Stock Prices")
            self.ax.set_xlabel("Date")
            self.ax.set_ylabel("Price")
            self.ax.legend()
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Update canvas
            self.canvas.draw()
            
        except requests.exceptions.HTTPError as http_err:
            messagebox.showerror("Error", f"HTTP error occurred: {http_err}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update chart: {str(e)}")

    def add_new_company(self):
        """
        Open a dialog to add a new company
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Company")
        dialog.geometry("400x200")
        
        # Company name
        ttk.Label(dialog, text="Company Name:").pack(pady=5)
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var).pack(fill="x", padx=5, pady=5)
        
        # Company symbol
        ttk.Label(dialog, text="Stock Symbol:").pack(pady=5)
        symbol_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=symbol_var).pack(fill="x", padx=5, pady=5)
        
        # Add button
        def add_company():
            try:
                # Search for company
                response = requests.get(f"{self.base_url}/search-company?query={name_var.get()}")
                response.raise_for_status()
                results = response.json()
                
                if not results:
                    messagebox.showerror("Error", "No company found matching the search")
                    return
                    
                # Add company
                company = results[0]  # Use the first matching company
                add_url = f"{self.base_url}/add-company"
                add_response = requests.post(add_url, json=company)
                add_response.raise_for_status()
                
                messagebox.showinfo("Success", "Company added successfully!")
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add company: {str(e)}")
        
        ttk.Button(dialog, text="Add Company", command=add_company).pack(pady=10)
        
    def refresh_data(self):
        """
        Refresh all data for the current company
        """
        if not self.current_symbol:
            messagebox.showwarning("Warning", "Please select a company first")
            return
            
        try:
            # Refresh stock data
            response = requests.get(f"{self.base_url}/predict-stock?symbol={self.current_symbol}")
            response.raise_for_status()
            result = response.json()
            
            # Update GUI
            self.price_label.config(text=f"Current Price: ${result['predicted_price']:.2f}")
            self.trend_label.config(text=f"Trend: {result['trend_prediction']} ({result['confidence']:.2f}% confidence)")
            
            # Update chart
            self.update_chart(self.current_symbol)
            
            messagebox.showinfo("Success", "Data refreshed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh data: {str(e)}")
            
    def update_data(self):
        """
        Periodically update stock data every 5 minutes
        """
        if self.current_symbol:
            try:
                # Fetch current price
                response = requests.get(f"{self.base_url}/predict-stock?symbol={self.current_symbol}")
                response.raise_for_status()
                result = response.json()
                
                # Update price
                self.price_label.config(text=f"Current Price: ${result['predicted_price']:.2f}")
                self.trend_label.config(text=f"Trend: {result['trend_prediction']} ({result['confidence']:.2f}% confidence)")
                
                # Update chart
                self.update_chart(self.current_symbol)
                
            except Exception as e:
                print(f"Error updating data: {str(e)}")
        
        # Schedule next update
        self.root.after(300000, self.update_data)  # 5 minutes in milliseconds
        
def main():
    root = tk.Tk()
    app = StockGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
