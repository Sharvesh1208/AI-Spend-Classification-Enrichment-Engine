// src/UserAvatar.jsx
function UserAvatar({ user, size = 40 }) {
  const getInitials = (name) => {
    if (!name) return 'U';
    const parts = name.split(' ');
    if (parts.length >= 2) {
      return (parts[0][0] + parts[1][0]).toUpperCase();
    }
    return name.substring(0, 2).toUpperCase();
  };

  return user.photoURL ? (
    <img
      src={user.photoURL}
      alt={user.displayName || user.email}
      style={{
        width: `${size}px`,
        height: `${size}px`,
        borderRadius: '50%',
        objectFit: 'cover',
        border: '2px solid #0070f3'
      }}
    />
  ) : (
    <div style={{
      width: `${size}px`,
      height: `${size}px`,
      borderRadius: '50%',
      background: 'linear-gradient(135deg, #0070f3, #003d82)',
      color: 'white',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontWeight: '600',
      fontSize: `${size / 2.5}px`,
      border: '2px solid #0070f3'
    }}>
      {getInitials(user.displayName || user.email)}
    </div>
  );
}

export default UserAvatar;