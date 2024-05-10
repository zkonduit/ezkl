use log::info;
use std::net::IpAddr;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use std::{fmt, pin::Pin};
#[doc(inline)]
pub use tokio_postgres::config::{
    ChannelBinding, Host, LoadBalanceHosts, SslMode, TargetSessionAttrs,
};
use tokio_postgres::tls::NoTlsStream;
use tokio_postgres::NoTls;
use tokio_postgres::{error::DbError, types::ToSql, Error, Row, Socket, ToStatement};

/// Connection configuration.
///
/// Configuration can be parsed from libpq-style connection strings. These strings come in two formats:
///
///
#[derive(Clone)]
pub struct Config {
    config: tokio_postgres::Config,
    notice_callback: Arc<dyn Fn(DbError) + Send + Sync>,
}

impl fmt::Debug for Config {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Config")
            .field("config", &self.config)
            .finish()
    }
}

impl Default for Config {
    fn default() -> Config {
        Config::new()
    }
}

impl Config {
    /// Creates a new configuration.
    pub fn new() -> Config {
        tokio_postgres::Config::new().into()
    }

    /// Sets the user to authenticate with.
    ///
    /// If the user is not set, then this defaults to the user executing this process.
    pub fn user(&mut self, user: &str) -> &mut Config {
        self.config.user(user);
        self
    }

    /// Gets the user to authenticate with, if one has been configured with
    /// the `user` method.
    pub fn get_user(&self) -> Option<&str> {
        self.config.get_user()
    }

    /// Sets the password to authenticate with.
    pub fn password<T>(&mut self, password: T) -> &mut Config
    where
        T: AsRef<[u8]>,
    {
        self.config.password(password);
        self
    }

    /// Gets the password to authenticate with, if one has been configured with
    /// the `password` method.
    pub fn get_password(&self) -> Option<&[u8]> {
        self.config.get_password()
    }

    /// Sets the name of the database to connect to.
    ///
    /// Defaults to the user.
    pub fn dbname(&mut self, dbname: &str) -> &mut Config {
        self.config.dbname(dbname);
        self
    }

    /// Gets the name of the database to connect to, if one has been configured
    /// with the `dbname` method.
    pub fn get_dbname(&self) -> Option<&str> {
        self.config.get_dbname()
    }

    /// Sets command line options used to configure the server.
    pub fn options(&mut self, options: &str) -> &mut Config {
        self.config.options(options);
        self
    }

    /// Gets the command line options used to configure the server, if the
    /// options have been set with the `options` method.
    pub fn get_options(&self) -> Option<&str> {
        self.config.get_options()
    }

    /// Sets the value of the `application_name` runtime parameter.
    pub fn application_name(&mut self, application_name: &str) -> &mut Config {
        self.config.application_name(application_name);
        self
    }

    /// Gets the value of the `application_name` runtime parameter, if it has
    /// been set with the `application_name` method.
    pub fn get_application_name(&self) -> Option<&str> {
        self.config.get_application_name()
    }

    /// Sets the SSL configuration.
    ///
    /// Defaults to `prefer`.
    pub fn ssl_mode(&mut self, ssl_mode: SslMode) -> &mut Config {
        self.config.ssl_mode(ssl_mode);
        self
    }

    /// Gets the SSL configuration.
    pub fn get_ssl_mode(&self) -> SslMode {
        self.config.get_ssl_mode()
    }

    /// Adds a host to the configuration.
    ///
    /// Multiple hosts can be specified by calling this method multiple times, and each will be tried in order. On Unix
    /// systems, a host starting with a `/` is interpreted as a path to a directory containing Unix domain sockets.
    /// There must be either no hosts, or the same number of hosts as hostaddrs.
    pub fn host(&mut self, host: &str) -> &mut Config {
        self.config.host(host);
        self
    }

    /// Gets the hosts that have been added to the configuration with `host`.
    pub fn get_hosts(&self) -> &[Host] {
        self.config.get_hosts()
    }

    /// Gets the hostaddrs that have been added to the configuration with `hostaddr`.
    pub fn get_hostaddrs(&self) -> &[IpAddr] {
        self.config.get_hostaddrs()
    }

    /// Adds a Unix socket host to the configuration.
    ///
    /// Unlike `host`, this method allows non-UTF8 paths.
    #[cfg(unix)]
    pub fn host_path<T>(&mut self, host: T) -> &mut Config
    where
        T: AsRef<Path>,
    {
        self.config.host_path(host);
        self
    }

    /// Adds a hostaddr to the configuration.
    ///
    /// Multiple hostaddrs can be specified by calling this method multiple times, and each will be tried in order.
    /// There must be either no hostaddrs, or the same number of hostaddrs as hosts.
    pub fn hostaddr(&mut self, hostaddr: IpAddr) -> &mut Config {
        self.config.hostaddr(hostaddr);
        self
    }

    /// Adds a port to the configuration.
    ///
    /// Multiple ports can be specified by calling this method multiple times. There must either be no ports, in which
    /// case the default of 5432 is used, a single port, in which it is used for all hosts, or the same number of ports
    /// as hosts.
    pub fn port(&mut self, port: u16) -> &mut Config {
        self.config.port(port);
        self
    }

    /// Gets the ports that have been added to the configuration with `port`.
    pub fn get_ports(&self) -> &[u16] {
        self.config.get_ports()
    }

    /// Sets the timeout applied to socket-level connection attempts.
    ///
    /// Note that hostnames can resolve to multiple IP addresses, and this timeout will apply to each address of each
    /// host separately. Defaults to no limit.
    pub fn connect_timeout(&mut self, connect_timeout: Duration) -> &mut Config {
        self.config.connect_timeout(connect_timeout);
        self
    }

    /// Gets the connection timeout, if one has been set with the
    /// `connect_timeout` method.
    pub fn get_connect_timeout(&self) -> Option<&Duration> {
        self.config.get_connect_timeout()
    }

    /// Sets the TCP user timeout.
    ///
    /// This is ignored for Unix domain socket connections. It is only supported on systems where
    /// TCP_USER_TIMEOUT is available and will default to the system default if omitted or set to 0;
    /// on other systems, it has no effect.
    pub fn tcp_user_timeout(&mut self, tcp_user_timeout: Duration) -> &mut Config {
        self.config.tcp_user_timeout(tcp_user_timeout);
        self
    }

    /// Gets the TCP user timeout, if one has been set with the
    /// `user_timeout` method.
    pub fn get_tcp_user_timeout(&self) -> Option<&Duration> {
        self.config.get_tcp_user_timeout()
    }

    /// Controls the use of TCP keepalive.
    ///
    /// This is ignored for Unix domain socket connections. Defaults to `true`.
    pub fn keepalives(&mut self, keepalives: bool) -> &mut Config {
        self.config.keepalives(keepalives);
        self
    }

    /// Reports whether TCP keepalives will be used.
    pub fn get_keepalives(&self) -> bool {
        self.config.get_keepalives()
    }

    /// Sets the amount of idle time before a keepalive packet is sent on the connection.
    ///
    /// This is ignored for Unix domain sockets, or if the `keepalives` option is disabled. Defaults to 2 hours.
    pub fn keepalives_idle(&mut self, keepalives_idle: Duration) -> &mut Config {
        self.config.keepalives_idle(keepalives_idle);
        self
    }

    /// Gets the configured amount of idle time before a keepalive packet will
    /// be sent on the connection.
    pub fn get_keepalives_idle(&self) -> Duration {
        self.config.get_keepalives_idle()
    }

    /// Sets the time interval between TCP keepalive probes.
    /// On Windows, this sets the value of the tcp_keepalive struct’s keepaliveinterval field.
    ///
    /// This is ignored for Unix domain sockets, or if the `keepalives` option is disabled.
    pub fn keepalives_interval(&mut self, keepalives_interval: Duration) -> &mut Config {
        self.config.keepalives_interval(keepalives_interval);
        self
    }

    /// Gets the time interval between TCP keepalive probes.
    pub fn get_keepalives_interval(&self) -> Option<Duration> {
        self.config.get_keepalives_interval()
    }

    /// Sets the maximum number of TCP keepalive probes that will be sent before dropping a connection.
    ///
    /// This is ignored for Unix domain sockets, or if the `keepalives` option is disabled.
    pub fn keepalives_retries(&mut self, keepalives_retries: u32) -> &mut Config {
        self.config.keepalives_retries(keepalives_retries);
        self
    }

    /// Gets the maximum number of TCP keepalive probes that will be sent before dropping a connection.
    pub fn get_keepalives_retries(&self) -> Option<u32> {
        self.config.get_keepalives_retries()
    }

    /// Sets the requirements of the session.
    ///
    /// This can be used to connect to the primary server in a clustered database rather than one of the read-only
    /// secondary servers. Defaults to `Any`.
    pub fn target_session_attrs(
        &mut self,
        target_session_attrs: TargetSessionAttrs,
    ) -> &mut Config {
        self.config.target_session_attrs(target_session_attrs);
        self
    }

    /// Gets the requirements of the session.
    pub fn get_target_session_attrs(&self) -> TargetSessionAttrs {
        self.config.get_target_session_attrs()
    }

    /// Sets the channel binding behavior.
    ///
    /// Defaults to `prefer`.
    pub fn channel_binding(&mut self, channel_binding: ChannelBinding) -> &mut Config {
        self.config.channel_binding(channel_binding);
        self
    }

    /// Gets the channel binding behavior.
    pub fn get_channel_binding(&self) -> ChannelBinding {
        self.config.get_channel_binding()
    }

    /// Sets the host load balancing behavior.
    ///
    /// Defaults to `disable`.
    pub fn load_balance_hosts(&mut self, load_balance_hosts: LoadBalanceHosts) -> &mut Config {
        self.config.load_balance_hosts(load_balance_hosts);
        self
    }

    /// Gets the host load balancing behavior.
    pub fn get_load_balance_hosts(&self) -> LoadBalanceHosts {
        self.config.get_load_balance_hosts()
    }

    /// Sets the notice callback.
    ///
    /// This callback will be invoked with the contents of every
    /// [`AsyncMessage::Notice`] that is received by the connection. Notices use
    /// the same structure as errors, but they are not "errors" per-se.
    ///
    /// Notices are distinct from notifications, which are instead accessible
    /// via the [`Notifications`] API.
    ///
    /// [`AsyncMessage::Notice`]: tokio_postgres::AsyncMessage::Notice
    /// [`Notifications`]: crate::Notifications
    pub fn notice_callback<F>(&mut self, f: F) -> &mut Config
    where
        F: Fn(DbError) + Send + Sync + 'static,
    {
        self.notice_callback = Arc::new(f);
        self
    }

    /// Opens a connection to a PostgreSQL database.
    pub async fn connect(&self) -> Result<Client, Error> {
        let (client, connection) = self.config.connect(NoTls).await?;

        let connection = Connection::new(connection);

        Ok(Client::new(client, connection))
    }
}

impl FromStr for Config {
    type Err = Error;

    fn from_str(s: &str) -> Result<Config, Error> {
        s.parse::<tokio_postgres::Config>().map(Config::from)
    }
}

impl From<tokio_postgres::Config> for Config {
    fn from(config: tokio_postgres::Config) -> Config {
        Config {
            config,
            notice_callback: Arc::new(|notice| {
                info!("{}: {}", notice.severity(), notice.message())
            }),
        }
    }
}

#[allow(missing_debug_implementations, dead_code)]
/// An asynchronous PostgreSQL connection. We use this to keep the connection alive / keep it pinned so that it doesn't
/// get dropped.
pub struct Connection {
    /// The underlying connection stream.
    connection: Pin<Box<tokio_postgres::Connection<Socket, NoTlsStream>>>,
}

impl Connection {
    /// Creates a new connection.
    pub fn new(connection: tokio_postgres::Connection<Socket, NoTlsStream>) -> Self {
        Connection {
            connection: Box::pin(connection),
        }
    }
}

#[allow(missing_debug_implementations, dead_code)]
/// An asynchronous PostgreSQL client.
pub struct Client {
    connection: Connection,
    client: tokio_postgres::Client,
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.close_inner();
    }
}

impl Client {
    pub(crate) fn new(client: tokio_postgres::Client, connection: Connection) -> Client {
        Client { client, connection }
    }

    /// A convenience function which parses a configuration string into a `Config` and then connects to the database.
    ///
    /// See the documentation for [`Config`] for information about the connection syntax.
    ///
    /// [`Config`]: config/struct.Config.html
    pub async fn connect(params: &str) -> Result<Client, Error> {
        params.parse::<Config>()?.connect().await
    }

    /// Returns a new `Config` object which can be used to configure and connect to a database.
    pub fn configure() -> Config {
        Config::new()
    }

    /// Executes a statement, returning the number of rows modified.
    ///
    /// A statement may contain parameters, specified by `$n`, where `n` is the index of the parameter of the list
    /// provided, 1-indexed.
    ///
    /// If the statement does not modify any rows (e.g. `SELECT`), 0 is returned.
    ///
    /// The `query` argument can either be a `Statement`, or a raw query string. If the same statement will be
    /// repeatedly executed (perhaps with different query parameters), consider preparing the statement up front
    /// with the `prepare` method.
    ///
    pub async fn execute<T>(
        &mut self,
        query: &T,
        params: &[&(dyn ToSql + Sync)],
    ) -> Result<u64, Error>
    where
        T: ?Sized + ToStatement,
    {
        self.client.execute(query, params).await
    }

    /// Executes a statement, returning the resulting rows.
    ///
    /// A statement may contain parameters, specified by `$n`, where `n` is the index of the parameter of the list
    /// provided, 1-indexed.
    ///
    /// The `query` argument can either be a `Statement`, or a raw query string. If the same statement will be
    /// repeatedly executed (perhaps with different query parameters), consider preparing the statement up front
    /// with the `prepare` method.
    ///
    /// # Examples
    ///
    pub async fn query<T>(
        &mut self,
        query: &T,
        params: &[&(dyn ToSql + Sync)],
    ) -> Result<Vec<Row>, Error>
    where
        T: ?Sized + ToStatement,
    {
        self.client.query(query, params).await
    }

    /// Determines if the client's connection has already closed.
    ///
    /// If this returns `true`, the client is no longer usable.
    pub fn is_closed(&self) -> bool {
        self.client.is_closed()
    }

    /// Closes the client's connection to the server.
    ///
    /// This is equivalent to `Client`'s `Drop` implementation, except that it returns any error encountered to the
    /// caller.
    pub fn close(mut self) -> Result<(), Error> {
        self.close_inner()
    }

    fn close_inner(&mut self) -> Result<(), Error> {
        self.client.__private_api_close();
        Ok(())
    }
}
