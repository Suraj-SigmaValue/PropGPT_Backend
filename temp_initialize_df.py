def initialize_dataframe(base_dir: Path):
    """Initialize combined dataframe from Excel (thread-safe)"""
    excel_path = base_dir / EXCEL_FILE
    pickle_path = base_dir / PICKLE_FILE
    
    # Acquire lock to prevent race conditions
    with _PICKLE_GENERATION_LOCK:
        logger.info("Acquired lock for pickle generation")
        
        # Check if another thread already generated the file while we were waiting
        if pickle_path.exists():
            try:
                logger.info("Pickle file exists, attempting to load...")
                df = joblib.load(pickle_path)
                df.columns = [normalize_colname(str(c)) for c in df.columns]
                logger.info(f"Pickle file loaded successfully by waiting thread. Shape: {df.shape}")
                return df
            except Exception as e:
                logger.warning(f"Pickle file exists but couldn't load: {e}. Regenerating...")
                try:
                    os.remove(pickle_path)
                except:
                    pass
        
        try:
            if not excel_path.exists():
                logger.error(f"Excel file not found: {excel_path}")
                return None
            
            logger.info(f"Loading data from {excel_path}...")
            dfs = pd.read_excel(excel_path, sheet_name=None)
            logger.info(f"Excel file loaded successfully")
            
            combined = []
            for ctype, cfg in SHEET_CONFIG.items():
                if cfg["sheet"] in dfs:
                    df = dfs[cfg["sheet"]].copy()
                    df.columns = [normalize_colname(str(c)) for c in df.columns]
                    df["__type"] = ctype
                    combined.append(df)
            
            if not combined:
                logger.error("No valid sheets found in Excel file!")
                return None
            
            df_all = pd.concat(combined, ignore_index=True)
            joblib.dump(df_all, pickle_path)
            logger.info(f"Pickle file saved successfully at {pickle_path}. Shape: {df_all.shape}")
            
            return df_all
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.exception("Data loading error")
            return None
