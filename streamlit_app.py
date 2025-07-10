import streamlit as st
import pandas as pd
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import traceback
import json
from io import BytesIO
import hashlib
import os

# 安全配置
ALLOWED_USERS = ["user1@company.com", "user2@company.com"]  # 允許的使用者清單
SESSION_TIMEOUT = 3600  # 1小時後自動登出

# 設定頁面配置
st.set_page_config(
    page_title="RFMTA 客戶分析工具 (安全版)",
    page_icon="🔐",
    layout="wide"
)

def check_authentication():
    """檢查使用者認證"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔐 使用者認證")
        
        # 從 secrets 讀取密碼
        try:
            correct_password = st.secrets["security"]["admin_password"]
        except:
            correct_password = "your_secure_password_here"  # 備用密碼
        
        password = st.text_input("請輸入存取密碼", type="password")
        
        if st.button("登入"):
            if password == correct_password:
                st.session_state.authenticated = True
                st.session_state.login_time = datetime.now()
                st.rerun()
            else:
                st.error("密碼錯誤")
        
        st.info("請聯繫管理員獲取存取密碼")
        return False
    
    # 檢查 session 是否過期
    if 'login_time' in st.session_state:
        elapsed = (datetime.now() - st.session_state.login_time).seconds
        if elapsed > SESSION_TIMEOUT:
            st.session_state.authenticated = False
            st.error("Session 已過期，請重新登入")
            st.rerun()
    
    return True

def sanitize_input(text):
    """清理使用者輸入"""
    if not text:
        return ""
    # 移除潛在的惡意字符
    import re
    return re.sub(r'[<>"\']', '', str(text))

class SecureRFMTAAnalyzer:
    def __init__(self):
        self.combined_data = None
        self.rfmt_result = None
        self.r_bins = None
        self.f_bins = None
        self.m_bins = None
        self.t_bins = None
        self.a_bins = None
        self.r_bounds = None
        self.f_bounds = None
        self.m_bounds = None
        self.t_bounds = None
        self.a_bounds = None
        self.frequency_by_sheet = None
        
    def create_google_sheet_output(self, export_df, sheet_title="RFMTA_Analysis"):
        """將結果輸出到新的 Google Sheet"""
        try:
            # 使用 Streamlit secrets 中的憑證（更安全）
            credentials_dict = st.secrets["google_credentials"]
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=['https://spreadsheets.google.com/feeds', 
                       'https://www.googleapis.com/auth/drive']
            )
            
            client = gspread.authorize(credentials)
            
            # 創建新的工作表，添加時間戳避免重複
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sheet_name = f"{sheet_title}_{timestamp}"
            
            # 創建新的 Google Sheet
            spreadsheet = client.create(sheet_name)
            worksheet = spreadsheet.sheet1
            
            # 設定工作表權限（可編輯但需要連結）
            spreadsheet.share(None, perm_type='anyone', role='writer')
            
            # 準備資料（轉換為字符串避免格式問題）
            data_to_write = []
            
            # 標題行
            headers = list(export_df.columns)
            data_to_write.append(headers)
            
            # 資料行
            for _, row in export_df.iterrows():
                row_data = []
                for col in headers:
                    value = row[col]
                    if pd.isna(value):
                        row_data.append("")
                    elif isinstance(value, (int, float)):
                        row_data.append(str(value))
                    else:
                        row_data.append(str(value))
                data_to_write.append(row_data)
            
            # 一次性寫入所有資料（提高效率）
            worksheet.update(data_to_write)
            
            # 格式化標題行
            worksheet.format("1:1", {
                "backgroundColor": {"red": 0.2, "green": 0.6, "blue": 0.9},
                "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}}
            })
            
            # 自動調整欄寬
            worksheet.columns_auto_resize(0, len(headers)-1)
            
            return spreadsheet.url, sheet_name
            
        except Exception as e:
            st.error(f"創建 Google Sheet 時發生錯誤: {str(e)}")
            return None, None
    
    @staticmethod
    def create_bins_and_labels(series, n_bins=4):
        """四分位分級函數"""
        min_val, max_val = series.min(), series.max()
        
        if min_val == max_val:
            bins = [min_val - 0.1, min_val + 0.1]
            labels = [1]
        else:
            quartiles = np.percentile(series.dropna(), np.linspace(0, 100, n_bins + 1))
            bins = np.unique(quartiles)
            
            if len(bins) < 4:
                bins = np.linspace(min_val, max_val + 1, n_bins + 1)
            
            labels = list(range(1, len(bins)))
        
        bins = np.unique(bins)
        print(f"✅ 最終使用的 bins for {series.name}: {bins}")
        return bins, labels, pd.Series(bins)
    
    def load_google_sheets_secure(self, sheet_names):
        """使用預設憑證安全載入 Google Sheets"""
        try:
            # 使用 Streamlit secrets 中的憑證
            credentials_dict = st.secrets["google_credentials"]
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=['https://spreadsheets.google.com/feeds', 
                       'https://www.googleapis.com/auth/drive']
            )
            
            client = gspread.authorize(credentials)
            
            self.data = []
            total_rows = 0
            
            for sheet_name in sheet_names:
                # 清理輸入
                sheet_name = sanitize_input(sheet_name)
                
                try:
                    sheet = client.open(sheet_name.strip()).sheet1
                    all_values = sheet.get_all_values()
                    
                    if not all_values:
                        st.warning(f"警告: {sheet_name} 是空的或無法讀取")
                        continue
                    
                    headers = all_values[0]
                    records = [dict(zip(headers, row)) for row in all_values[1:]]
                    df = pd.DataFrame(records)
                    df['SheetSource'] = sheet_name
                    df['OriginalRow'] = range(2, len(df) + 2)
                    
                    self.data.append(df)
                    total_rows += len(df)
                    st.success(f"已載入 {len(df)} 筆資料從 {sheet_name}")
                    
                except Exception as e:
                    st.error(f"處理 {sheet_name} 時發生錯誤: {str(e)}")
            
            if not self.data:
                raise ValueError("沒有資料可以從任何工作表載入")
            
            self.combined_data = pd.concat(self.data, ignore_index=True)
            
            # 篩選有效數據
            self.combined_data = self.combined_data[
                (self.combined_data['Email'].notna()) & 
                (self.combined_data['Email'] != '') & 
                (self.combined_data['付款狀態'] == '已付款') &
                (pd.to_numeric(self.combined_data['實際付款金額'], errors='coerce') > 0)
            ]
            
            # 將 Email 轉換為小寫
            self.combined_data['Email'] = self.combined_data['Email'].str.lower()
            
            # 解析日期
            self.combined_data['訂單時間'] = pd.to_datetime(
                self.combined_data['訂單時間'], 
                errors='coerce'
            )
            
            st.success(f"資料載入完成！總共 {len(self.combined_data)} 筆有效記錄")
            
            return True
            
        except Exception as e:
            st.error(f"載入資料時發生錯誤: {str(e)}")
            return False
    
    def analyze_rfmta(self):
        """執行 RFMTA 分析"""
        try:
            if self.combined_data is None:
                st.error("請先載入資料")
                return False
            
            st.info("開始 RFMTA 分析...")
            
            now = datetime.now().date()
            email_col = 'Email'
            order_date_col = '訂單時間'
            total_amount_col = '實際付款金額'
            name_col = '姓名'
            
            # 確保金額欄位被正確解析為數值
            self.combined_data[total_amount_col] = pd.to_numeric(
                self.combined_data[total_amount_col], 
                errors='coerce'
            ).fillna(0)
            
            # 診斷日期解析問題
            date_na_count = self.combined_data[order_date_col].isna().sum()
            if date_na_count > 0:
                st.warning(f"警告：有 {date_na_count} 筆訂單時間無法解析")
            
            # 計算 F（跨作品參與次數）
            frequency_by_sheet = self.combined_data.groupby(['Email', 'SheetSource']).size().reset_index(name='sheet_frequency')
            frequency_by_sheet['sheet_frequency'] = 1
            total_frequency = frequency_by_sheet.groupby('Email')['sheet_frequency'].sum().reset_index()
            
            # 計算 T（總參與次數）
            total_times = self.combined_data.groupby('Email').size().reset_index(name='total_times')
            
            # 計算 M（總金額）
            total_monetary = self.combined_data.groupby(email_col)[total_amount_col].sum().reset_index()
            
            # 選擇最佳姓名
            latest_orders = self.combined_data.sort_values(order_date_col).groupby('Email').last().reset_index()
            latest_orders['has_participant'] = latest_orders[name_col].str.contains('參加者', na=False)
            
            name_selection = []
            for email, group in latest_orders.groupby('Email'):
                non_participant_names = group.loc[~group['has_participant'], name_col]
                if not non_participant_names.empty:
                    name_selection.append((email, non_participant_names.iloc[0]))
                else:
                    name_selection.append((email, group[name_col].iloc[0]))
            
            name_selection_df = pd.DataFrame(name_selection, columns=['Email', 'preferred_name'])
            
            # 初始化 RFMT DataFrame
            rfmt = pd.DataFrame()
            
            # 計算 Recency
            rfmt['Recency'] = self.combined_data.groupby(email_col)[order_date_col].max().apply(
                lambda x: (now - pd.to_datetime(x).date()).days if pd.notnull(x) else None
            )
            
            # R 分層
            rfmt['R'], self.r_bins = pd.qcut(
                rfmt['Recency'],
                q=4,
                labels=False,
                retbins=True,
                duplicates='drop'
            )
            
            num_labels = len(np.unique(rfmt['R'].dropna()))
            rfmt['R'] = num_labels - rfmt['R']
            rfmt['R'] = rfmt['R'].fillna(-1).astype(int)
            
            # 檢查無效 R 值
            r_neg1_count = (rfmt['R'] == -1).sum()
            if r_neg1_count > 0:
                st.warning(f"警告：有 {r_neg1_count} 位客戶因訂單時間解析失敗而被標記為 R=-1")
            
            # 計算其他指標
            rfmt['Frequency'] = total_frequency.set_index('Email')['sheet_frequency']
            rfmt['Monetary'] = total_monetary.set_index('Email')[total_amount_col]
            rfmt['Times'] = total_times.set_index('Email')['total_times']
            rfmt['Average'] = rfmt['Monetary'] / rfmt['Times']
            rfmt['Name'] = name_selection_df.set_index('Email')['preferred_name']

            # ============ 修改的 F 計算邏輯開始 ============
            # F 的固定分級：1作品=F1, 2作品=F2, 3作品=F3, 4+作品=F4
            def calculate_f_score(frequency):
                """
                根據參加作品數計算 F 分數
                F1 = 參加 1 個作品
                F2 = 參加 2 個作品  
                F3 = 參加 3 個作品
                F4 = 參加 4 個或以上作品
                """
                if frequency == 1:
                    return 1
                elif frequency == 2:
                    return 2
                elif frequency == 3:
                    return 3
                elif frequency >= 4:
                    return 4
                else:
                    return 1  # 預設值，理論上不會發生

            # 應用新的 F 計算邏輯
            rfmt['F'] = rfmt['Frequency'].apply(calculate_f_score)

            # 設定 F 的邊界值（用於輸出說明）
            self.f_bins = [0.5, 1.5, 2.5, 3.5, float('inf')]  # 分界點
            self.f_bounds = pd.Series(self.f_bins)  # 保持與原版格式一致
            # ============ 修改的 F 計算邏輯結束 ============

            # M, T, A 四分位分層（保持原來的邏輯）
            self.m_bins, m_labels, self.m_bounds = self.create_bins_and_labels(rfmt['Monetary'])
            self.t_bins, t_labels, self.t_bounds = self.create_bins_and_labels(rfmt['Times'])
            self.a_bins, a_labels, self.a_bounds = self.create_bins_and_labels(rfmt['Average'])

            # 注意：這裡不再計算 F，因為已經在上面用固定分級了
            rfmt['M'] = pd.cut(rfmt['Monetary'], bins=self.m_bins, labels=m_labels, include_lowest=True)
            rfmt['T'] = pd.cut(rfmt['Times'], bins=self.t_bins, labels=t_labels, include_lowest=True)
            rfmt['A'] = pd.cut(rfmt['Average'], bins=self.a_bins, labels=a_labels, include_lowest=True)
            
            # RFMTA 組合
            rfmt['RFMTA_Score'] = (rfmt['R'].astype(str) + rfmt['F'].astype(str) + 
                                 rfmt['M'].astype(str) + rfmt['T'].astype(str) + 
                                 rfmt['A'].astype(str))
            
            self.rfmt_result = rfmt
            self.frequency_by_sheet = frequency_by_sheet
            
            st.success("RFMTA 分析完成！")
            return True
            
        except Exception as e:
            st.error(f"分析時發生錯誤: {str(e)}")
            return False
    
    def get_export_data(self):
        """準備匯出資料"""
        if self.rfmt_result is None:
            return None
        
        try:
            # 重新獲取完整資料進行匯出
            # 由於我們保留了 frequency_by_sheet，可以重建需要的資訊
            
            # 基本的 RFMTA 結果
            export_df = self.rfmt_result.copy()
            export_df = export_df.reset_index()
            
            # 重新命名 Name 為 姓名
            if 'Name' in export_df.columns:
                export_df = export_df.rename(columns={'Name': '姓名'})
            
            # 基本輸出欄位
            base_columns = ['Email', '姓名', 'Recency', 'Frequency', 'Monetary', 'Times', 'Average',
                           'R', 'F', 'M', 'T', 'A', 'RFMTA_Score']
            
            # 確保所有基本欄位都存在
            for col in base_columns:
                if col not in export_df.columns:
                    export_df[col] = ""
            
            # 作品參與次數欄位（從原始資料重建）
            if hasattr(self, 'combined_data') and self.combined_data is not None:
                sheet_columns = sorted(set(self.combined_data['SheetSource'].unique()))
                
                # 填充每個作品的參與次數
                for sheet in sheet_columns:
                    participation_counts = self.combined_data[
                        self.combined_data['SheetSource'] == sheet
                    ].groupby('Email').size()
                    export_df[sheet] = export_df['Email'].map(participation_counts).fillna(0).astype(int)
            else:
                sheet_columns = []
            
            # 準備邊界說明欄位
            boundary_columns = []
            
            # ============ R 的邊界說明 ============
            if hasattr(self, 'r_bins') and self.r_bins is not None:
                for i in range(len(self.r_bins)-1):
                    idx = len(self.r_bins) - 2 - i  # 反轉索引
                    next_idx = idx + 1
                    export_df[f'R{i+1}_Boundary'] = f"R{i+1}: {self.r_bins[idx]:.0f} to less than {self.r_bins[next_idx]:.0f} days"
                    boundary_columns.append(f'R{i+1}_Boundary')
            
            # ============ F 的邊界說明（修改後的固定分級）============
            f_boundary_descriptions = [
                "F1: participated in 1 different work",
                "F2: participated in 2 different works", 
                "F3: participated in 3 different works",
                "F4: participated in 4 or more different works"
            ]
            
            for i, description in enumerate(f_boundary_descriptions):
                export_df[f'F{i+1}_Boundary'] = description
                boundary_columns.append(f'F{i+1}_Boundary')
            
            # ============ M 的邊界說明 ============
            if hasattr(self, 'm_bins') and self.m_bins is not None:
                for i in range(len(self.m_bins)-1):
                    if i == len(self.m_bins)-2:
                        export_df[f'M{i+1}_Boundary'] = f"M{i+1}: ${self.m_bins[i]:.0f} and above"
                    else:
                        next_value = self.m_bins[i+1]
                        export_df[f'M{i+1}_Boundary'] = f"M{i+1}: ${self.m_bins[i]:.0f} to less than ${next_value:.0f}"
                    boundary_columns.append(f'M{i+1}_Boundary')
            
            # ============ T 的邊界說明 ============
            if hasattr(self, 't_bins') and self.t_bins is not None:
                for i in range(len(self.t_bins)-1):
                    if i == len(self.t_bins)-2:
                        export_df[f'T{i+1}_Boundary'] = f"T{i+1}: {int(self.t_bins[i])} times or more in total"
                    else:
                        current_value = int(self.t_bins[i]) + 1 if i > 0 else int(self.t_bins[i])
                        next_value = int(self.t_bins[i+1])
                        export_df[f'T{i+1}_Boundary'] = f"T{i+1}: {current_value} to {next_value} times in total"
                    boundary_columns.append(f'T{i+1}_Boundary')
            
            # ============ A 的邊界說明 ============
            if hasattr(self, 'a_bins') and self.a_bins is not None:
                for i in range(len(self.a_bins)-1):
                    if i == len(self.a_bins)-2:
                        export_df[f'A{i+1}_Boundary'] = f"A{i+1}: ${self.a_bins[i]:.0f} and above per participation"
                    else:
                        next_value = self.a_bins[i+1]
                        export_df[f'A{i+1}_Boundary'] = f"A{i+1}: ${self.a_bins[i]:.0f} to less than ${next_value:.0f} per participation"
                    boundary_columns.append(f'A{i+1}_Boundary')
            
            # 組合最終的欄位順序
            columns_to_export = base_columns + sheet_columns + boundary_columns
            
            # 確保所有指定的列都存在，若不存在則填充為空
            for col in columns_to_export:
                if col not in export_df.columns:
                    export_df[col] = ""
            
            return export_df[columns_to_export].copy()
            
        except Exception as e:
            st.error(f"準備匯出資料時發生錯誤: {str(e)}")
            return None

# Streamlit 應用程式主體
def main():
    # 檢查認證
    if not check_authentication():
        return
    
    st.title("🔐 RFMTA 客戶分析工具 (安全版)")
    st.markdown("---")
    
    # 顯示登入資訊
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🚪 登出"):
            st.session_state.authenticated = False
            st.rerun()
    
    # 初始化分析器
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SecureRFMTAAnalyzer()
    
    # 側邊欄 - 資料載入
    st.sidebar.header("🔧 設定")
    
    # Google Sheets 名稱輸入
    st.sidebar.subheader("輸入 Google Sheets 名稱")
    sheet_names_input = st.sidebar.text_area(
        "請輸入工作表名稱（每行一個）",
        height=100,
        placeholder="例如：\n2024年活動報名表\n2023年會員名單"
    )
    
    # 載入資料按鈕
    if st.sidebar.button("🔄 載入資料", type="primary"):
        if sheet_names_input:
            sheet_names = [sanitize_input(name.strip()) for name in sheet_names_input.split('\n') if name.strip()]
            
            with st.spinner("載入資料中..."):
                success = st.session_state.analyzer.load_google_sheets_secure(sheet_names)
                if success:
                    st.rerun()
        else:
            st.sidebar.error("請輸入工作表名稱")
    
    # 主要內容區域
    if st.session_state.analyzer.combined_data is not None:
        # 資料概覽（不顯示敏感資訊）
        st.subheader("📋 資料概覽")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("總記錄數", len(st.session_state.analyzer.combined_data))
        
        with col2:
            unique_customers = st.session_state.analyzer.combined_data['Email'].nunique()
            st.metric("唯一客戶數", unique_customers)
        
        with col3:
            unique_sheets = st.session_state.analyzer.combined_data['SheetSource'].nunique()
            st.metric("作品數量", unique_sheets)
        
        # 顯示作品參與統計
        if st.checkbox("顯示作品參與統計"):
            sheet_counts = st.session_state.analyzer.combined_data['SheetSource'].value_counts()
            st.write("**各作品參與統計:**")
            for sheet, count in sheet_counts.items():
                st.write(f"- {sheet}: {count} 筆記錄")
        
        # 分析按鈕
        if st.button("🔍 執行 RFMTA 分析", type="primary"):
            with st.spinner("分析中..."):
                success = st.session_state.analyzer.analyze_rfmta()
                if success:
                    st.rerun()
    
    # 顯示分析結果
    if st.session_state.analyzer.rfmt_result is not None:
        st.subheader("📈 RFMTA 分析結果")
        
        # 分析結果統計
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**各指標分布**")
            for metric in ['R', 'F', 'M', 'T', 'A']:
                dist = st.session_state.analyzer.rfmt_result[metric].value_counts().sort_index()
                st.write(f"{metric}: {dict(dist)}")
        
        with col2:
            st.write("**RFMTA Score 範例**")
            sample_scores = st.session_state.analyzer.rfmt_result['RFMTA_Score'].head(5)
            for score in sample_scores.values:
                st.write(f"Score: {score}")
        
        # 顯示完整結果表格
        if st.checkbox("顯示完整分析結果"):
            # 不顯示敏感的 Email 和姓名資訊
            display_columns = ['Recency', 'Frequency', 'Monetary', 'Times', 'Average', 'R', 'F', 'M', 'T', 'A', 'RFMTA_Score']
            available_columns = [col for col in display_columns if col in st.session_state.analyzer.rfmt_result.columns]
            st.dataframe(st.session_state.analyzer.rfmt_result[available_columns])
        
        # 創建 Google Sheet 輸出
        st.subheader("📊 創建分析結果 Google Sheet")
        
        output_title = st.text_input("輸出檔案名稱", value="RFMTA_Analysis")
        
        if st.button("📝 創建 Google Sheet", type="primary"):
            with st.spinner("正在創建 Google Sheet..."):
                export_data = st.session_state.analyzer.get_export_data()
                
                if export_data is not None:
                    sheet_url, sheet_name = st.session_state.analyzer.create_google_sheet_output(
                        export_data, 
                        sanitize_input(output_title)
                    )
                    
                    if sheet_url:
                        st.success("✅ Google Sheet 創建成功！")
                        st.markdown(f"**📋 工作表名稱:** {sheet_name}")
                        st.markdown(f"**🔗 [點擊這裡開啟 Google Sheet]({sheet_url})**")
                        
                        # 分享指引
                        st.info("""
                        **分享說明:**
                        - 此 Google Sheet 已設定為「任何有連結的人都可以編輯」
                        - 您可以直接分享上方連結給同事
                        - 如需更改權限，請在 Google Sheet 中點擊右上角「分享」按鈕
                        """)
                        
                        # 顯示匯出欄位說明
                        with st.expander("📋 匯出欄位說明"):
                            st.markdown("""
                            **基本欄位:**
                            - Email, 姓名, Recency, Frequency, Monetary, Times, Average
                            - R, F, M, T, A, RFMTA_Score
                            
                            **作品參與次數:**
                            - 各作品的個別參與次數統計
                            
                            **邊界說明:**
                            - R1-R4_Boundary: 最近購買時間分級說明
                            - F1-F4_Boundary: 參與作品數分級說明（固定分級）
                            - M1-M4_Boundary: 消費金額分級說明
                            - T1-T4_Boundary: 總參與次數分級說明
                            - A1-A4_Boundary: 平均消費分級說明
                            
                            **F 分級特色（已修改為固定分級）:**
                            - F1: 參加 1 個作品
                            - F2: 參加 2 個作品
                            - F3: 參加 3 個作品
                            - F4: 參加 4 個或以上作品
                            """)
    
    else:
        if st.session_state.analyzer.combined_data is None:
            st.info("👆 請先在左側載入資料")
            
            # 使用說明
            st.subheader("📖 使用說明")
            st.markdown("""
            **安全版本特色:**
            - 🔐 需要密碼認證才能使用
            - 🔒 使用預設的 Google API 憑證，無需上傳敏感檔案
            - 🗑️ 分析完成後自動清除原始資料
            - ⏱️ Session 會在 1 小時後自動過期
            - 📊 結果直接輸出為 Google Sheet，易於分享
            
            **使用步驟:**
            1. 在左側輸入要分析的 Google Sheets 名稱
            2. 點擊「載入資料」
            3. 執行 RFMTA 分析
            4. 創建結果 Google Sheet 並分享
            
            **RFMTA 分析說明:**
            - **R (Recency)**: 最近購買時間（四分位數分級）
            - **F (Frequency)**: 跨作品參與次數（固定分級：1,2,3,4+個作品）
            - **M (Monetary)**: 總消費金額（四分位數分級）
            - **T (Times)**: 總參與次數（四分位數分級）
            - **A (Average)**: 平均每次消費金額（四分位數分級）
            """)

if __name__ == "__main__":
    main()
