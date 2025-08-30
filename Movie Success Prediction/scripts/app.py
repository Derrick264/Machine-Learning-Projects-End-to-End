import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('models/rf_best_model.pkl')

st.title("Movie Success Prediction")

# Integer inputs (step=1)
color = st.number_input('color', min_value=0, max_value=10, step=1, value=1)
director_name = st.number_input('director_name', min_value=0, step=1, value=79)
actor_2_name = st.number_input('actor_2_name', min_value=0, step=1, value=104)
actor_1_name = st.number_input('actor_1_name', min_value=0, step=1, value=155)
actor_3_name = st.number_input('actor_3_name', min_value=0, step=1, value=0)
plot_keywords = st.number_input('plot_keywords', min_value=0, step=1, value=123456)  # Example encoded int
aspect_ratio_missing = st.number_input('aspect_ratio_missing', min_value=0, max_value=1, step=1, value=0)

# Float inputs
num_critic_for_reviews = st.number_input('num_critic_for_reviews', value=723.0)
duration = st.number_input('duration', value=178.0)
director_facebook_likes = st.number_input('director_facebook_likes', value=0.0)
actor_3_facebook_likes = st.number_input('actor_3_facebook_likes', value=855.0)
actor_1_facebook_likes = st.number_input('actor_1_facebook_likes', value=1000.0)
num_voted_users = st.number_input('num_voted_users', value=886204.0)
cast_total_facebook_likes = st.number_input('cast_total_facebook_likes', value=4834.0)
facenumber_in_poster = st.number_input('facenumber_in_poster', value=0.0)
num_user_for_reviews = st.number_input('num_user_for_reviews', value=3054.0)
language = st.number_input('language', value=11.0)
country = st.number_input('country', value=62.0)
content_rating = st.number_input('content_rating', value=7.0)
budget = st.number_input('budget', value=200000000.0)
title_year = st.number_input('title_year', value=2009.0)
actor_2_facebook_likes = st.number_input('actor_2_facebook_likes', value=936.0)
imdb_score = st.number_input('imdb_score', value=7.9)
aspect_ratio = st.number_input('aspect_ratio', value=1.78)
movie_facebook_likes = st.number_input('movie_facebook_likes', value=33000.0)
log_gross = st.number_input('log_gross', value=19.622)
log_budget = st.number_input('log_budget', value=19.114)
log_cast_total_facebook_likes = st.number_input('log_cast_total_facebook_likes', value=8.484)
log_num_voted_users = st.number_input('log_num_voted_users', value=13.695)
roi = st.number_input('roi', value=1.662)
star_power = st.number_input('star_power', value=2791.0)
movie_age = st.number_input('movie_age', value=16.0)

# Genre binary flags (0 or 1)
action = st.number_input('action', min_value=0, max_value=1, step=1, value=1)
adventure = st.number_input('adventure', min_value=0, max_value=1, step=1, value=1)
animation = st.number_input('animation', min_value=0, max_value=1, step=1, value=0)
biography = st.number_input('biography', min_value=0, max_value=1, step=1, value=0)
comedy = st.number_input('comedy', min_value=0, max_value=1, step=1, value=0)
crime = st.number_input('crime', min_value=0, max_value=1, step=1, value=0)
documentary = st.number_input('documentary', min_value=0, max_value=1, step=1, value=0)
drama = st.number_input('drama', min_value=0, max_value=1, step=1, value=0)
family = st.number_input('family', min_value=0, max_value=1, step=1, value=0)
fantasy = st.number_input('fantasy', min_value=0, max_value=1, step=1, value=0)
film_noir = st.number_input('film-noir', min_value=0, max_value=1, step=1, value=0)
game_show = st.number_input('game-show', min_value=0, max_value=1, step=1, value=0)
history = st.number_input('history', min_value=0, max_value=1, step=1, value=0)
horror = st.number_input('horror', min_value=0, max_value=1, step=1, value=0)
music = st.number_input('music', min_value=0, max_value=1, step=1, value=0)
musical = st.number_input('musical', min_value=0, max_value=1, step=1, value=0)
mystery = st.number_input('mystery', min_value=0, max_value=1, step=1, value=0)
news = st.number_input('news', min_value=0, max_value=1, step=1, value=0)
reality_tv = st.number_input('reality-tv', min_value=0, max_value=1, step=1, value=0)
romance = st.number_input('romance', min_value=0, max_value=1, step=1, value=0)
sci_fi = st.number_input('sci-fi', min_value=0, max_value=1, step=1, value=0)
short = st.number_input('short', min_value=0, max_value=1, step=1, value=0)
sport = st.number_input('sport', min_value=0, max_value=1, step=1, value=0)
thriller = st.number_input('thriller', min_value=0, max_value=1, step=1, value=0)
war = st.number_input('war', min_value=0, max_value=1, step=1, value=0)
western = st.number_input('western', min_value=0, max_value=1, step=1, value=0)

# Prepare input dataframe with all features
input_df = pd.DataFrame({
    'color': [color],
    'director_name': [director_name],
    'num_critic_for_reviews': [num_critic_for_reviews],
    'duration': [duration],
    'director_facebook_likes': [director_facebook_likes],
    'actor_3_facebook_likes': [actor_3_facebook_likes],
    'actor_2_name': [actor_2_name],
    'actor_1_facebook_likes': [actor_1_facebook_likes],
    'gross': [0.0],  # You can provide 0 or a default numeric value if needed
    'actor_1_name': [actor_1_name],
    'num_voted_users': [num_voted_users],
    'cast_total_facebook_likes': [cast_total_facebook_likes],
    'actor_3_name': [actor_3_name],
    'facenumber_in_poster': [facenumber_in_poster],
    'plot_keywords': [plot_keywords],
    'num_user_for_reviews': [num_user_for_reviews],
    'language': [language],
    'country': [country],
    'content_rating': [content_rating],
    'budget': [budget],
    'title_year': [title_year],
    'actor_2_facebook_likes': [actor_2_facebook_likes],
    'imdb_score': [imdb_score],
    'aspect_ratio': [aspect_ratio],
    'movie_facebook_likes': [movie_facebook_likes],
    'action': [action],
    'adventure': [adventure],
    'animation': [animation],
    'biography': [biography],
    'comedy': [comedy],
    'crime': [crime],
    'documentary': [documentary],
    'drama': [drama],
    'family': [family],
    'fantasy': [fantasy],
    'film-noir': [film_noir],
    'game-show': [game_show],
    'history': [history],
    'horror': [horror],
    'music': [music],
    'musical': [musical],
    'mystery': [mystery],
    'news': [news],
    'reality-tv': [reality_tv],
    'romance': [romance],
    'sci-fi': [sci_fi],
    'short': [short],
    'sport': [sport],
    'thriller': [thriller],
    'war': [war],
    'western': [western],
    'log_gross': [log_gross],
    'log_budget': [log_budget],
    'log_cast_total_facebook_likes': [log_cast_total_facebook_likes],
    'log_num_voted_users': [log_num_voted_users],
    'roi': [roi],
    'star_power': [star_power],
    'movie_age': [movie_age],
    'aspect_ratio_missing': [aspect_ratio_missing]
})

if st.button('Predict Movie Success'):
    prediction = model.predict(input_df)[0]
    st.success(f'Predicted Success Category: {prediction}')
