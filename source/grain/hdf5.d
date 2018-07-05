/**
  HDF5 bindings

  This file is part of HDF5.  The full HDF5 copyright notice, including     *
  terms governing use, modification, and redistribution, is contained in    *
  the files COPYING and Copyright.html.  COPYING can be found at the root   *
  of the source code distribution tree; Copyright.html can be found at the  *
  root level of an installed copy of the electronic HDF5 document set and   *
  is linked from the top-level documents page.  It can also be found at     *
  http://hdfgroup.org/HDF5/doc/Copyright.html.  If you do not have          *
  access to either file, you may request a copy from help@hdfgroup.org.     *
  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  Ported to D by Laeeth Isharc 2014
  Borrowed heavily in terms of C API declarations from https://github.com/SFrijters/hdf5-d
  Stefan Frijters bindings for D

  Bindings probably not yet complete or bug-free.

  Consider this not even alpha stage.  It probably isn't so far away from being useful though.
  This is written for Linux and will need modification to work on other platforms.

  Modified by Shigeki Karita 2018
 */
module grain.hdf5;

extern (C):

alias hid_t = int;
alias hsize_t = ulong;
alias hssize_t = long;
alias herr_t = int;

/** Define atomic datatypes */
enum H5S_ALL = 0;
///
enum H5S_UNLIMITED = (cast(hsize_t)cast(hssize_t)(-1));
/** Define user-level maximum number of dimensions */
enum H5S_MAX_RANK = 32;



// this ddoc make adrdox die

// absence of rdwr => rd-only
enum H5F_ACC_RDONLY  = 0x0000u;
// open for read and write
enum H5F_ACC_RDWR    = 0x0001u;
// overwrite existing files
enum H5F_ACC_TRUNC   = 0x0002u;
// fail if file already exists
enum H5F_ACC_EXCL    = 0x0004u;
// print debug info
enum H5F_ACC_DEBUG   = 0x0008u;
// create non-existing files
enum H5F_ACC_CREAT   = 0x0010u;

// Value passed to H5Pset_elink_acc_flags to cause flags to be taken from the parent file.
enum H5F_ACC_DEFAULT = 0xffffu; /*ignore setting on lapl     */


// Default value for all property list classes 
enum H5P_DEFAULT = 0;


/* The IEEE floating point types in various byte orders. */
alias H5T_IEEE_F32BE = H5T_IEEE_F32BE_g;
alias H5T_IEEE_F32LE = H5T_IEEE_F32LE_g;
alias H5T_IEEE_F64BE = H5T_IEEE_F64BE_g;
alias H5T_IEEE_F64LE = H5T_IEEE_F64LE_g;
extern __gshared hid_t H5T_IEEE_F32BE_g;
extern __gshared hid_t H5T_IEEE_F32LE_g;
extern __gshared hid_t H5T_IEEE_F64BE_g;
extern __gshared hid_t H5T_IEEE_F64LE_g;

/*
 * These are "standard" types.  For instance, signed (2's complement) and
 * unsigned integers of various sizes and byte orders.
 */
alias H5T_STD_I8BE = H5T_STD_I8BE_g;
alias H5T_STD_I8LE = H5T_STD_I8LE_g;
alias H5T_STD_I16BE = H5T_STD_I16BE_g;
alias H5T_STD_I16LE = H5T_STD_I16LE_g;
alias H5T_STD_I32BE = H5T_STD_I32BE_g;
alias H5T_STD_I32LE = H5T_STD_I32LE_g;
alias H5T_STD_I64BE = H5T_STD_I64BE_g;
alias H5T_STD_I64LE = H5T_STD_I64LE_g;
alias H5T_STD_U8BE = H5T_STD_U8BE_g;
alias H5T_STD_U8LE = H5T_STD_U8LE_g;
alias H5T_STD_U16BE = H5T_STD_U16BE_g;
alias H5T_STD_U16LE = H5T_STD_U16LE_g;
alias H5T_STD_U32BE = H5T_STD_U32BE_g;
alias H5T_STD_U32LE = H5T_STD_U32LE_g;
alias H5T_STD_U64BE = H5T_STD_U64BE_g;
alias H5T_STD_U64LE = H5T_STD_U64LE_g;
alias H5T_STD_B8BE = H5T_STD_B8BE_g;
alias H5T_STD_B8LE = H5T_STD_B8LE_g;
alias H5T_STD_B16BE = H5T_STD_B16BE_g;
alias H5T_STD_B16LE = H5T_STD_B16LE_g;
alias H5T_STD_B32BE = H5T_STD_B32BE_g;
alias H5T_STD_B32LE = H5T_STD_B32LE_g;
alias H5T_STD_B64BE = H5T_STD_B64BE_g;
alias H5T_STD_B64LE = H5T_STD_B64LE_g;
alias H5T_STD_REF_OBJ = H5T_STD_REF_OBJ_g;
alias H5T_STD_REF_DSETREG = H5T_STD_REF_DSETREG_g;
extern __gshared hid_t H5T_STD_I8BE_g;
extern __gshared hid_t H5T_STD_I8LE_g;
extern __gshared hid_t H5T_STD_I16BE_g;
extern __gshared hid_t H5T_STD_I16LE_g;
extern __gshared hid_t H5T_STD_I32BE_g;
extern __gshared hid_t H5T_STD_I32LE_g;
extern __gshared hid_t H5T_STD_I64BE_g;
extern __gshared hid_t H5T_STD_I64LE_g;
extern __gshared hid_t H5T_STD_U8BE_g;
extern __gshared hid_t H5T_STD_U8LE_g;
extern __gshared hid_t H5T_STD_U16BE_g;
extern __gshared hid_t H5T_STD_U16LE_g;
extern __gshared hid_t H5T_STD_U32BE_g;
extern __gshared hid_t H5T_STD_U32LE_g;
extern __gshared hid_t H5T_STD_U64BE_g;
extern __gshared hid_t H5T_STD_U64LE_g;
extern __gshared hid_t H5T_STD_B8BE_g;
extern __gshared hid_t H5T_STD_B8LE_g;
extern __gshared hid_t H5T_STD_B16BE_g;
extern __gshared hid_t H5T_STD_B16LE_g;
extern __gshared hid_t H5T_STD_B32BE_g;
extern __gshared hid_t H5T_STD_B32LE_g;
extern __gshared hid_t H5T_STD_B64BE_g;
extern __gshared hid_t H5T_STD_B64LE_g;
extern __gshared hid_t H5T_STD_REF_OBJ_g;
extern __gshared hid_t H5T_STD_REF_DSETREG_g;

///
hid_t  H5Fcreate(const char *filename, uint flags, hid_t create_plist, hid_t access_plist);
///
hid_t  H5Fopen(const char *filename, uint flags, hid_t access_plist);
///
herr_t H5Fclose(hid_t file_id);

///
hid_t H5Screate_simple(int rank, const hsize_t *dims, const hsize_t *maxdims);
///
herr_t H5Sclose(hid_t space_id);

///
hid_t H5Dopen2(hid_t file_id, const char *name, hid_t dapl_id);
///
hid_t H5Dcreate2(hid_t loc_id, const char *name, hid_t type_id,
                 hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id);//
///
herr_t H5Dclose(hid_t dset_id);
///
herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t plist_id, void *buf/*out*/);
///
herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t plist_id, const void *buf);
